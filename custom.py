import numpy as np
import sympy
from scipy.optimize import LinearConstraint, Bounds, milp

class mat_gen:
    def __init__(self, grid_size = 3, symbolic = 1, eps_value = 1, walls = 3, random_seed = 0):
        
        np.random.seed(seed = random_seed)
        
        self.N_line = grid_size
        self.N = self.N_line * self.N_line
        self.action_num = 4

        # if we use epsilon or use a number
        self.symbolic = symbolic

        # Pi is the optimal policy, which includes 
        # 0: going down, 1: going right, 2: going up, and 3: going left (4 directions)
        self.Pi = np.random.randint(0, 4, size=self.N)
        # Pi = np.random.randint(2, 3, size=N)
        # Define the transition matrix and the action matrix

        if symbolic == 1:
            self.TransMatrix = np.zeros((self.N, self.N), dtype=object)
            self.TransMatrix_3d = np.zeros((self.N, self.N, self.action_num), dtype=object)
            self.ActionMatrix = np.zeros((self.N, self.action_num), dtype=object)
        else:
            self.TransMatrix = np.zeros((self.N, self.N))
            self.TransMatrix_3d = np.zeros((self.N, self.N, self.action_num))
            self.ActionMatrix = np.zeros((self.N, self.action_num))
        # ---------
        # [8, 7, 6]
        # [5, 4, 3]
        # [2, 1, 0]
        # ---------
        self.Walls = np.zeros((self.N, 4))
        self.Walls[:self.N_line, 0] = 1
        for i in range(self.N_line):
            self.Walls[i*self.N_line, 1] = 1
        self.Walls[-self.N_line:, 2] = 1
        for i in range(self.N_line):
            self.Walls[i*self.N_line+self.N_line-1, 3] = 1
        # generate walls
        add_walls_num = walls
        added_walls_num = 0
        while(1):
            a_idx = np.random.randint(0, self.N)
            a_dir = np.random.randint(0, self.action_num)
            if self.Walls[a_idx, a_dir] == 0:
                if a_dir == 0:
                    b_idx = a_idx - self.N_line
                    b_dir = 2
                elif a_dir == 1:
                    b_idx = a_idx - 1
                    b_dir = 3
                elif a_dir == 2:
                    b_idx = a_idx + self.N_line
                    b_dir = 0
                else:
                    b_idx = a_idx + 1
                    b_dir = 1
                
                if sum(self.Walls[a_idx, :]) < 3 and sum(self.Walls[b_idx, :]) < 3:
                    # print(sum(self.Walls[a_idx, :]))
                    # print(sum(self.Walls[b_idx, :]))
                    self.Walls[a_idx, a_dir] = 1
                    self.Walls[b_idx, b_dir] = 1
                    
                    added_walls_num += 1
            
            if added_walls_num == add_walls_num:
                break
            
        # Define the epsilon
        if symbolic == 1:
            self.e = sympy.symbols('e')
        else:
            self.e = eps_value
    def action_mat(self):
        for i in range(self.N):
            for j in range(self.action_num):
                if j == self.Pi[i]:
                    self.ActionMatrix[i, j] = self.e + (1-self.e) / self.action_num
                else:
                    self.ActionMatrix[i, j] = (1-self.e) / self.action_num
        return self.ActionMatrix
    def transition_mat(self):   
        # Calculate transition matrix
        for i in range(self.N):
            for j in range(self.action_num):
                if self.Pi[i] == j:
                    add_num = self.e + (1 - self.e) / self.action_num
                else:
                    add_num = (1 - self.e) / self.action_num
                    
                if self.Walls[i, j] == 1:
                    self.TransMatrix[i, i] += add_num
                else:
                    if j == 0:
                        b_idx = i - self.N_line
                    elif j == 1:
                        b_idx = i - 1
                    elif j == 2:
                        b_idx = i + self.N_line
                    elif j == 3:
                        b_idx = i + 1
                    self.TransMatrix[i, b_idx] += add_num
        return self.TransMatrix
    def get_actions(self, previous = 0, current = 0):   
        # Calculate transition matrix
        for i in range(self.N):
            for j in range(self.action_num):
                if self.Pi[i] == j:
                    add_num = self.e + (1 - self.e) / self.action_num
                else:
                    add_num = (1 - self.e) / self.action_num
                    
                if self.Walls[i, j] == 1:
                    self.TransMatrix_3d[i, i, j] = add_num
                else:
                    if j == 0:
                        b_idx = i - self.N_line
                    elif j == 1:
                        b_idx = i - 1
                    elif j == 2:
                        b_idx = i + self.N_line
                    elif j == 3:
                        b_idx = i + 1
                    self.TransMatrix_3d[i, b_idx, j] = add_num
        return self.TransMatrix_3d[previous, current, :]

class const_gen:
    def __init__(self, grid_size, sim_time, action_size, p_init, mat_cls, preferred, not_preferred):
        self.P = preferred
        self.Q = not_preferred
        self.S = grid_size**2
        self.T = sim_time
        self.mat_cls = mat_cls
        self.transition = self.mat_cls.transition_mat()
        self.p_init = p_init
        
    def equality_const(self):
        beq = np.zeros(shape = (self.T*self.S, 1))
        beq[0:self.S,0] = self.p_init
        Aeq = np.zeros(shape = (self.T*self.S, self.T*self.S*4+2))

        col_cunter = 0
        for i in range(self.T):
            for j in range(self.S):
                Aeq[i*self.S+ j, col_cunter:col_cunter+4]=[1,1,1,1]
                col_cunter = col_cunter + 4
            if i>0:
                for j in range(self.S):
                    current = j
                    previous_list = np.where(self.transition[:,current])[0]
                    for k, previous in enumerate(previous_list):
                        actions_list = np.where(self.mat_cls.get_actions(previous = previous, current = current))[0]
                        for l, action in enumerate(actions_list):
                            Aeq[i*self.S+ j, (i-1)*self.S*4+previous*4+action] = -1 * self.transition[previous, current]  
        return Aeq, beq
    
    def inequality_const(self):
        eps = 1e-3
        b = np.zeros(shape = (6, 1))
        A = np.zeros(shape = (6, self.T*self.S*4+2))
        
        Plist = []
        for i, p in enumerate(self.P):
            x = (self.T-1)*self.S*4 + p*4
            Plist += [x, x+1, x+2, x+3]
        
        Qlist = []
        for i, q in enumerate(self.Q):
            x = (self.T-1)*self.S*4 + q*4
            Qlist += [x, x+1, x+2, x+3]
        
        V_id = self.T*self.S*4
        B_id = self.T*self.S*4 + 1
        #  V-B<0
        A[0, V_id] = 1
        A[0, B_id] = -1
        b[0] = 0
        # -V<0
        A[1, V_id] = -1
        b[1] = 0
        # -P+V<0
        A[2, V_id] = 1
        b[2] = 0
        for i, p in enumerate(Plist):
            A[2, p] = -1
        # P-V+B<1
        for i, p in enumerate(Plist):
            A[3, p] = 1
        A[3, V_id] = -1
        A[3, B_id] = 1
        b[3] = 1
        # P-Q-B(1+eps)<-eps
        for i, p in enumerate(Plist):
            A[4, p] = 1
        for i, q in enumerate(Qlist):
            A[4, q] = -1
        A[4, B_id] = -1*(1+eps)
        b[4] = -1 * eps
        # Q-P+B(1+eps)<1
        for i, p in enumerate(Plist):
            A[5, p] = -1
        for i, q in enumerate(Qlist):
            A[5, q] = 1
        A[5, B_id] = 1+eps
        b[5] = 1 
        
        return A, b
    
    def bounds(self):
        
        lb = np.zeros(shape = (self.T*self.S*4+2, 1))
        ub = np.ones(shape = (self.T*self.S*4+2, 1))
        
        return lb, ub 
    
    def intcon(self):
        
        B_id = self.T*self.S*4 + 1
        
        return B_id
    
    def objective_func(self):
        
        f = np.zeros(shape = (1, self.T*self.S*4+2))
        f[0,self.T*self.S*4] = -1
        
        return f
    
class solve_milp:   
    def __init__(self, Aeq, A, beq, b, f, intcon):
        self.A = np.concatenate((Aeq, A))
        self.bl = np.concatenate((beq, -np.inf*np.ones(shape = b.shape)))[:,0]
        self.bu = np.concatenate((beq, b))[:,0]
        self.c = f.T[:,0]
        self.intcon = np.zeros_like(self.c)
        self.intcon[intcon] = 1
        
    def solve(self):
        constraints = LinearConstraint(self.A, self.bl, self.bu)
        bounds = Bounds(lb=0, ub=1, keep_feasible=False)
        res = milp(c=self.c, constraints=constraints, bounds= bounds, integrality=self.intcon)
        
        return res.x
        
        
        