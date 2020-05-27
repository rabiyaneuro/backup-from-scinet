# -*- coding: utf-8 -*-
"""
Created on Feb 8

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

version_num = 8
delays_mat = np.load("Anat Data/delays_mat{}g.npy".format(version_num))
w_mat = np.load("Anat Data/w_mat{}g.npy".format(version_num))
ue_array = np.load("Anat Data/ue_array{}g.npy".format(version_num))

#%% TARGET DATA 

nodes = 10
num_dim = int((((nodes**2)-nodes)/2))

skip = 200
targ_data = hf.plot_cor_mat(ue_array, nodes, skip)
np.fill_diagonal(targ_data, 0)
#%% DIFF EVOLUTION PARAMS

bounds = []
lower = 1
upper = 5
for n in range(num_dim):
    bounds.append((lower,upper))
    
evol_params= {
        'strategy': 'best2bin',
        'maxiter': 500,
        'popsize': 15,
        'tol': 0.5,
        'mut': (0.5, 1),
        'recomb': 0.9,
        'polish': False,
        'init': 'latinhypercube',
        'atol':0,
        'mse' : 0.005,
        'optim': 'd',
        'bounds': bounds,
        'heavi': False,
        'prior': None,
        'scaleP': True
        } #tryingt to optimize conduc velocity 'c' or weights 'w' or delays 'd'?

#%% WILSON-COWAN PARAMS 

"""Set seed for the wc_model_sim in the residuals fxn (so all potential 
solutions get tested with same noise variable)"""
wc_seed = 2000

wc_params = {
        'c1': 1.6,
        'c2': -4.7,
        'c3': 3,
        'c4': -0.63,
        'I_e': 1.8,
        'I_i': -0.2,
        'g': 0.08,
        'time_steps': 2000,
        'dt': 0.01,
        'd': 0.001,
        'constant': True
        }

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