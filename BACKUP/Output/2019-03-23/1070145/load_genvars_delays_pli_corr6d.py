# -*- coding: utf-8 -*-
"""
Created on Feb 20

load_genvars_delaysTEMPLATE.py

This file is a template for loading the parameters needed for the simulation when optimizaing
delays using generated data

for modelling the possible solutions we will be using 

hf.wcm.wc_modelsim_c(wc_params, tract_mat, c_mat, nodes,
                                       seed_num = network_seed)

objective function will use both pli and correlation

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
c_mat = np.ones((nodes,nodes))*10000
c_mat = np.reciprocal(c_mat)
np.fill_diagonal(c_mat,0)

#simulating network
#WILSON-COWAN PARAMS 

"""Set seed for the wc_model_sim in the residuals fxn (so all potential 
solutions get tested with same noise variable)"""
wc_seed = 40
targ_seed = 0
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
        'd': 0.001,
        'constant': True
        }

ue_array, ui_array, delays = hf.wcm.wc_model_sim_new(wc_params, tract_mat, c_mat, w_mat, nodes,
                                       seed_num = targ_seed)

#correlation matrix
skip = 200
targ_data_corr = hf.plot_cor_mat(ue_array, nodes, skip)
np.fill_diagonal(targ_data_corr, 0)

#PLI
import mne

#signal properties
_Dt = wc_params['dt']
_alpha = 10
_dt = _Dt/_alpha # time step is 1ms: _dt = 0.001 #0.002
fs = 1/_dt # Sampling rate, or number of measurements per second

chunk = 600
method = "wpli"
fmin = 13
fmax = 30

ue_targ_chunks= []
num_chunks = int(1800/chunk)

for ch in range(num_chunks):
    ue_targ_chunks.append(ue_array[:,skip:][:,ch*chunk:ch*chunk + chunk])


_con, _freqs, _times, _n_epochs, _n_tapers = mne.connectivity.spectral_connectivity(
    data=ue_targ_chunks,method=method, sfreq=fs, fmin = fmin, fmax = fmax, faverage = True,
    mode = 'fourier', verbose= True)


targ_data_pli = _con[:,:,0]

#%% DIFF EVOLUTION PARAMS

bounds = []

lower = 500
upper = 10000

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

args = (wc_params, fs, nodes, targ_data_corr, targ_data_pli, w_mat, chunk, skip, wc_seed, evol_params['optim'], tract_mat, evol_params['heavi'], fmin, fmax)