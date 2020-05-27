#%%
import time
import helper_functions as hf
import differentialevolution_par_scinet as df
import numpy as np

def make_symmetric(mat):
    n_mat = np.ndarray.copy(mat)
    for r in range(n_mat.shape[0]):
        for c in range(n_mat.shape[0]):
            n_mat[c,r] = n_mat[r,c]
    return n_mat

nodes = 10

wc_seed = 0

version_num = 1
tract_mat = np.load("Anat Data/tract_mat{}g.npy".format(version_num))
tract_mat= make_symmetric(tract_mat)

w_mat = np.ones((nodes,nodes))
np.fill_diagonal(w_mat,0)

#cv matrix
c_mat = np.ones((nodes, nodes))*4000
c_mat = np.reciprocal(c_mat)


#%% WILSON-COWAN PARAMS 

"""Set seed for the wc_model_sim in the residuals fxn (so all potential 
solutions get tested with same noise variable)"""

wc_params = {
        'c1': 1.6,
        'c2': -4.7,
        'c3': 3,
        'c4': -0.63,
        'I_e': 1.8,
        'I_i': -0.2,
        'g': 0.05,
        'time_steps': 2000,
        'dt': 0.01,
        'd': 0,
        'constant': True
        }
#%% TARGET DATA 
num_dim = int((((nodes**2)-nodes)/2))
ue_array, ui_array, delays = hf.wcm.wc_model_sim_new(wc_params, tract_mat, c_mat, w_mat, nodes,
                                       seed_num = wc_seed)
#correlation matrix
skip = 200
targ_data = hf.plot_cor_mat(ue_array, nodes, skip)
np.fill_diagonal(targ_data, 0)


#%% DIFF EVOLUTION PARAMS

bounds = []
lower = 3000
upper = 7000
for n in range(num_dim):
    bounds.append((lower,upper))


evol_params= {
        'strategy': 'best1bin',
        'maxiter': 400,
        'popsize': 15,
        'tol': 0.01,
        'mut': (0.5, 1),
        'recomb': 0.9,
        'polish': False,
        'init': 'latinhypercube',
        'atol':0,
        'mse' : 0.001,
        'optim': 'c',
        'bounds': bounds,
        'heavi': False,
        'prior': None,
        'scaleP': True
        } #tryingt to optimize conduc velocity 'c' or weights 'w' or delays 'd'?



#%% ARGS FOR RESIDUAL FXN IN DIFF EVOLUTION ALROGITHM
"""residuals_cw_corr(c, params, nodes, targ_data, cw_mat, skip, seed, optim, tract_mat=None, heavyside =False,
                   plot = None):"""

if evol_params['optim'] =='w':
    MAT = c_mat
    args = (wc_params, nodes, targ_data, MAT, skip, wc_seed, evol_params['optim'], 
            tract_mat, evol_params['heavi'], None)

elif evol_params['optim'] =='c':
    MAT = w_mat
    args = (wc_params, nodes, targ_data, MAT, skip, wc_seed, evol_params['optim'], tract_mat, evol_params['heavi'], None)
    
elif evol_params['optim'] =='d':
    MAT = w_mat
    args = (wc_params, nodes, targ_data, MAT, skip, wc_seed, evol_params['optim'], None,
            evol_params['heavi'], None)