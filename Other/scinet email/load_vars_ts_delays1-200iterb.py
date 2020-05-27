# -*- coding: utf-8 -*-
"""
Created on Dec  11

This file loads the parameters needed for the simulation when optimizaing
delays using the time series from sick kids dataset

@author: Rabiya Noori
"""

#%% Import modules

import time
import helper_functions as hf
import differentialevolution_par_scinet as df
import numpy as np

#%% Setting paramaters
file_num = "ST09"
#Set seed for the wc_model_sim in the residuals fxn
wc_seed = 0

nodes = 10
num_dim = int((((nodes**2)-nodes)/2))

# WILSON-COWAN PARAMS 
wc_params = {
        'c1': 1.6,
        'c2': -4.7,
        'c3': 3,
        'c4': -0.63,
        'I_e': 1.8,
        'I_i': -0.2,
        'g': 0.08,
        'cond_vel': 4000,
        'time_steps': 2000,
        'dt': 0.01,
        'd': 0.001
        }

# DIFF EVOLUTION PARAMS
evol_params= {
        'strategy': 'best2bin',
        'maxiter': 200,
        'popsize': 15,
        'tol': 0.5,
        'mut': 0.5,
        'recomb': 0.7,
        'polish': False,
        'init': 'latinhypercube',
        'atol':0,
        'mse' : 0,
        'bound_l': 0,
        'bound_u':4000
        }

# SIGNAL PROPERTIES
_Dt = wc_params['dt']
_alpha = 10
_dt = _Dt/_alpha # time step is 1ms: _dt = 0.001
fs = 1/_dt # Sampling rate, or number of measurements per second

# WEIGHTS AND TRACT

# sick kids

catmatrix = np.load("Anat Data/catmatrix{}.npy".format(file_num))
tract_mat = np.load("Anat Data/tract_mat{}_r.npy".format(file_num))


# TARGET DATA
all_ts = []
for i in range(10):
    all_ts.append(catmatrix[i,3,:])

skip = 200
targ_data = hf.plot_cor_mat(np.array(all_ts), nodes, skip)
np.fill_diagonal(targ_data, 0)

# ARGS FOR RESIDUAL FXN IN DIFF EVOLUTION ALROGITHM
args = (wc_params, targ_data, nodes, tract_mat, skip, wc_seed)