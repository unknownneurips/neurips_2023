from custom import mat_gen, const_gen, solve_milp
import numpy as np

random_seed = 7 # Policy generation, initial position, wall placement
grid_size = 3
goal = [6]
# P > Q 
P = [5,7,8]
Q = [state for state in np.arange(0,grid_size**2) if state not in P]
# --3*3----
# [8, 7, 6]
# [5, 4, 3]
# [2, 1, 0]
# ---------
# --4*4----
# [15, 14, 13, 12]
# [11, 10, 09, 08]
# [07, 06, 05, 04]
# [03, 02, 01, 00]
# ---------

# 0: down, 1: right, 2: up, and 3: left

symbolic = 0
eps_value = 0.8
walls = 2
T = 3
S = grid_size**2
A = 4 

np.random.seed(seed = random_seed)
rnd = np.random.rand(grid_size**2) 
p_init = rnd / sum(rnd)        

inst = mat_gen(grid_size = grid_size, symbolic = symbolic, eps_value = eps_value, 
               walls = walls, random_seed=random_seed)

# act = inst.get_actions(previous = 0, current = 1)
transition = inst.TransMatrix


const = const_gen(grid_size = grid_size, sim_time = T, action_size = A, p_init = p_init, mat_cls = inst, preferred = P, not_preferred = Q)
Aeq, beq = const.equality_const()
A, b = const.inequality_const()
lb, ub = const.bounds()
intcon = const.intcon()
f = const.objective_func()  

# Aeq[4,0] = 0.5*Aeq[4,0]
# Aeq[4,1] = 0.5*Aeq[4,1]

# Aeq[5,4] = 0.5*Aeq[5,4]
# Aeq[5,7] = 0.5*Aeq[5,7]

# Aeq[6,9] = 0.5*Aeq[6,9]
# Aeq[6,10] = 0.5*Aeq[6,10]

# Aeq[7,14] = 0.5*Aeq[7,14]
# Aeq[7,15] = 0.5*Aeq[7,15]


solver = solve_milp(Aeq, A, beq, b, f, intcon)
x = solver.solve()


if x is None:
    vps = 0
else: 
    vps = x[len(x)-2]
print('value of preference satisfaction is = ', vps)

ytable = np.reshape(x[:-2], newshape = (-1, 4))

# for i in range(T):
#     print(f'sum of p in T={i} equals to {np.sum(ytable[i*(grid_size**2):(i+1)*(grid_size**2),:])}')
# goal_chance = 0
# for j in goal:
#     for i in range(T):
#         ind = i*(grid_size**2)+ j
#         goal_chance = goal_chance + sum(ytable[ind])

# sat = np.zeros(shape = (T,1))
# for i in range(T):
#     chance = 0
#     for j in P:
#         ind = i*(grid_size**2)+ j
#         chance = chance + sum(ytable[ind])
#     sat[i] = chance
# np.save('ytable.npy', ytable)



                


                       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    