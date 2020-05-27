# -*- coding: utf-8 -*-
"""
Created on Feb 20

load_genvars_delaysTEMPLATE.py

This file is a template for loading the parameters needed for the simulation when optimizaing
delays using generated data

for modelling the possible solutions we will be using 

hf.wcm.wc_modelsim_c(wc_params, tract_mat, c_mat, nodes,
                                       seed_num = network_seed)

hf.residuals_corr

used by run_delay_scinet_alt.py

@author: Rabiya Noori
"""
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

nodes= 10
num_dim = int((((nodes**2)-nodes)/2))

seed_tract = 123
rng = np.random.RandomState(seed_tract)
t_mat_v = rng.uniform(low = 1,high=50, size=(num_dim))
tract_mat = hf.p2matrix(t_mat_v, nodes)

# constant w_mat matches where tract_mat is 0
w_mat = np.ones((nodes,nodes))
zero_ind = tract_mat==0
w_mat[zero_ind] =0
np.fill_diagonal(w_mat,0)

#cv matrix
c_mat = np.ones((nodes,nodes))*4000
c_mat = np.reciprocal(c_mat)
np.fill_diagonal(c_mat,0)

#simulating network
#WILSON-COWAN PARAMS 

"""Set seed for the wc_model_sim in the residuals fxn (so all potential 
solutions get tested with same noise variable)"""
wc_seed = 0
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

ue_array, ui_array, delays = hf.wcm.wc_model_sim_new(wc_params, tract_mat, c_mat, w_mat, nodes,
                                       seed_num = wc_seed)
#correlation matrix
skip = 200
targ_data = hf.plot_cor_mat(ue_array, nodes, skip)
np.fill_diagonal(targ_data, 0)
#%% DIFF EVOLUTION PARAMS

bounds = []

lower = 4000
upper = 7000

for n in range(num_dim):
    bounds.append((lower,upper))
    
evol_params= {
        'strategy': 'best1bin',
        'maxiter': 1000,
        'popsize': 15,
        'tol': 0.01,
        'mut': (0.5,1),
        'recomb': 0.9,
        'polish': False,
        'init': 'latinhypercube',
        'atol':0,
        'mse' : 0,
        'optim': 'c',
        'bounds': bounds,
        'heavi': False,
        'prior': None,
        'scaleP': True
        }
#%% ARGS FOR RESIDUAL FXN IN DIFF EVOLUTION ALROGITHM
"""def residuals_cw_corr(c, params, nodes, targ_data, cw_mat, skip, seed, optim, tract_mat=None, heavyside =False,
                   plot = None)

residuals_cw_corr_sparse(c, params, nodes, targ_data, cw_mat, skip, seed, optim, tract_mat, heavyside =False):"""

if evol_params['optim'] =='c':
    MAT = w_mat
    args = (wc_params, nodes, targ_data, MAT, skip, wc_seed, evol_params['optim'], tract_mat, evol_params['heavi'])