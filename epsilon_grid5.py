from custom import mat_gen, const_gen, solve_milp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

grid_size = 5
goal = 15
# P > Q 
P = [3,4,8,9]
Q = [state for state in np.arange(0,grid_size**2) if state not in P]
walls = 10
T = 10 
random_seed = 8 # 97, 89, 85, 80, 73, 63, 59, 51, 36, 24, 8
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
# --5*5----
# [24, 23, 22, 21, 20] 
# [19, 18, 17, 16, 15] 
# [14, 13, 12, 11, 10] 
# [09, 08, 07, 06, 05] 
# [04, 03, 02, 01, 00]
# ---------
# --8*8----
# [63, 62, 61, 60, 59, 58, 57, 56]
# [55, 54, 53, 52, 51, 50, 49, 48]
# [47, 46, 45, 44, 43, 42, 41, 40] 
# [39, 38, 37, 36, 35, 34, 33, 32]
# [31, 30, 29, 28, 27, 26, 25, 24]
# [23, 22, 21, 20, 19, 18, 17, 16]
# [15, 14, 13, 12, 11, 10, 09, 08]
# [07, 06, 05, 04, 03, 02, 01, 00]
# --10*10----
# -----------
# [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
# [89, 88, 87, 86, 85, 84, 83, 82, 81, 80]
# [79, 78, 77, 76, 75, 74, 73, 72, 71, 70]
# [69, 68, 67, 66, 65, 64, 63, 62, 61, 60]
# [59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
# [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
# [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
# [29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
# [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
# [09, 08, 07, 06, 05, 04, 03, 02, 01, 00]
# ---------
# 0: down, 1: right, 2: up, and 3: left

symbolic = 0
S = grid_size**2
A = 4 

np.random.seed(seed = random_seed)
rnd = np.random.rand(grid_size**2) 
p_init = rnd / sum(rnd)        

step = 0.01
eps_list = np.arange(start = 0,stop = 1 + step, step = step)
eps_list = [0.8]

vps_record = []
goal_record = []
for eps_value in eps_list:
    
    inst = mat_gen(grid_size = grid_size, symbolic = symbolic, eps_value = eps_value, 
                   walls = walls, random_seed=random_seed)
    
    const = const_gen(grid_size = grid_size, sim_time = T, action_size = A, p_init = p_init, mat_cls = inst, preferred = P, not_preferred = Q)
    Aeq, beq = const.equality_const()
    A, b = const.inequality_const()
    lb, ub = const.bounds()
    intcon = const.intcon()
    f = const.objective_func()  
    
    solver = solve_milp(Aeq, A, beq, b, f, intcon)
    x = solver.solve()
    
    if x is None:
        vps = 0
        goal_record.append(0)
    else: 
        vps = x[len(x)-2]
        ytable = np.reshape(x[:-2], newshape = (-1, 4))
        goal_chance = 0
        for i in range(T):
            ind = i*(grid_size**2)+ goal
            goal_chance = goal_chance + sum(ytable[ind])
        goal_record.append(goal_chance)
        
    vps_record.append(vps)
    # print(f'value of preference satisfaction is = {vps} for epsilon = {eps_value}')
    
np.save(f'table_{grid_size}.npy',ytable)

goal_record = np.reshape(goal_record, newshape = (-1,1))
scaler = MinMaxScaler(feature_range=(0,max(vps_record))) 
# scaler = StandardScaler() 

goal_record = scaler.fit_transform(goal_record)
plt.plot([1, 1], [0.0, 0.197], linestyle='dashed', linewidth=2, color='grey')
plt.plot([0.0, 1.0], [0.197, 0.197], linestyle='dashed', linewidth=2, color='grey')
plt.plot(eps_list, vps_record, label = r'$exploration\;(v_{ps})$', linewidth = 3)
plt.plot(eps_list, goal_record, label = r'$exploitation\;(minmax\; scaled)$', linewidth = 3)
plt.plot([1], [0.197], 'o', markersize=10, color='green')
plt.legend(fontsize = 12)
plt.xlabel(r'$\epsilon$', fontsize = 16)
plt.ylabel(r'$probability$', fontsize = 16)
plt.tick_params(labelsize = 16)
plt.title(f'random_seed = {random_seed}', fontsize = 16)
# plt.ylabel('value of preference satisfaction')
plt.grid(True)
plt.savefig(f'Results/{grid_size}/random_seed_{random_seed}.png',dpi = 600, bbox_inches='tight')
plt.show()
