# -*- coding: utf-8 -*-
"""
Created on July 31, 2019

Runs optimization to find delays in a network using the Kuramoto model

@author: Administrator
"""
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
import differentialevolution_par_scinet_discrete as df
import numpy as np

def make_symmetric(mat):
    n_mat = np.ndarray.copy(mat)
    for r in range(n_mat.shape[0]):
        for c in range(n_mat.shape[0]):
            n_mat[c,r] = n_mat[r,c]
    return n_mat

print("starting...")
nodes= 10
num_dim = int((((nodes**2)-nodes)/2))
lower_b = 5
upper_b = 125

lower_d =5
upper_d =5

# constant w_mat matches where tract_mat is 0
w_mat = np.ones((nodes,nodes))
np.fill_diagonal(w_mat,0)

#delay matrix
#d_mat = np.ones((nodes,nodes),dtype=int)*upper
#np.fill_diagonal(d_mat,0)
#
seed_tract = 123
rng = np.random.RandomState(seed_tract)
d_mat_v = rng.uniform(low = lower_d,high=upper_d, size=num_dim)
d_mat = hf.p2matrix(d_mat_v, nodes).astype(int)
np.fill_diagonal(d_mat,0)

d_mat = make_symmetric(d_mat)

#
#simulating network
#KURAMOTO PARAMS 

"""Set seed for the wc_model_sim in the residuals fxn (so all potential 
solutions get tested with same noise variable)"""
km_seed = 0
wi = np.ones((10,1))*5
km_params = {
        'wi': wi, 
        'K': 1,
        'N': 10,
        'time_steps': 2000,
        'dt': 0.01,
        'constant': True
        }

print("generating target data...")
ue_array = hf.wcm.km_model_sim_d(km_params, d_mat, upper_b, w_mat, nodes)

#targ data

targ_data = ue_array
#%% DIFF EVOLUTION PARAMS

bounds = []

for n in range(num_dim):
    bounds.append((lower_b,upper_b))
    
evol_params= {
        'strategy': 'best1bin',
        'maxiter': 400,
        'popsize': 11,
        'tol': -10,
        'mut': 0.4,
        'recomb': 0.9,
        'polish': False,
        'init': 'latinhypercube',
        'atol':0,
        'mse' : 0,
        'optim': 'd',
        'bounds': bounds,
        'heavi': False,
        'prior': None,
        'scaleP': True,
        'res_metric': "time"
        }
# ARGS FOR RESIDUAL FXN IN DIFF EVOLUTION ALROGITHM
args = (km_params, nodes, targ_data, w_mat, upper_b, evol_params['res_metric'])