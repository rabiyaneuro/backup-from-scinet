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

version_num = 5
c_mat = np.load("Anat Data/c_mat{}g.npy".format(version_num))
tract_mat = np.load("Anat Data/tract_mat{}g.npy".format(version_num))
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
lower = 0
upper = 1
for n in range(num_dim):
    bounds.append((lower,upper))
    
evol_params= {
        'strategy': 'best2bin',
        'maxiter': 400,
        'popsize': 15,
        'tol': 0.5,
        'mut': 0.5,
        'recomb': 0.7,
        'polish': False,
        'init': 'latinhypercube',
        'atol':0,
        'mse' : 0.005,
        'optim': 'w',
        'bounds': bounds,
        'heavi': False
        } #tryingt to optimize conduc velocity 'c' or weights 'w'?

#%% WILSON-COWAN PARAMS 

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
        'g': 0.08,
        'time_steps': 2000,
        'dt': 0.01,
        'd': 0,
        'constant': True
        }
#%% ARGS FOR RESIDUAL FXN IN DIFF EVOLUTION ALROGITHM
"""def residuals_cw_corr(c, params, targ_data, nodes, tract_mat, cw_mat, skip, seed, optim, heavyside =False,
                   plot = None):"""
if evol_params['optim'] =='w':
    MAT = c_mat
elif evol_params['optim'] =='c':
    MAT = w_mat
args = (wc_params, targ_data, nodes, tract_mat, MAT, skip, wc_seed, evol_params['optim'], evol_params['heavi'], False)