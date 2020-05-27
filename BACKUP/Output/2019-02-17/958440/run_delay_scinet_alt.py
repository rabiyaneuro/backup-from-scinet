# -*- coding: utf-8 -*-
"""
Created on Feb 8

This is for running on SciNet
This file sets parameters for the WC brain scale model- computes the correlation
using randomly generated anatomically data or importing from cocomaq. It runs the
optimization to find the best delays to generate the system which has a correlation
that best matches the target

Jan 31- Updating the file so it receives the name of the variable script from 
command line and imports it accordingly

Feb 8 - This is the same as ren_delay_scinet but uses a diff modelling function in 
residuals so it accepts c_mat AND a w_mat (for use with generated data)

@author: Administrator
"""
#%% Import modules
import sys
from mpi4py import MPI

variable_script = sys.argv[1]
job = sys.argv[2]
exec(open(variable_script).read()) #loading variables into the environment

#initialize the MPI interface
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#%% Seed
"""initialize random num gen for each process of diff evolution- 
so that the choosing of solution vectors is randomized on each node"""
rng = np.random.RandomState(rank+1)
#%% RUN OPTIMIZATION

prior = []
for p in range(evol_params['popsize']):
    prior.append(hf.matrix2p(targ_data))
    
start_time = time.time()
res = df.differential_evolution(hf.residuals_cw_corr, args = args,
                              strategy = evol_params['strategy'],
                              popsize = evol_params['popsize'],
                              mutation = evol_params['mut'],
                              recombination = evol_params['recomb'],
                              bounds = evol_params['bounds'],
                              tol = evol_params['tol'],
                              maxiter = evol_params['maxiter'],
                              seed = rng,
                              disp = True,
                              polish = evol_params['polish'],
                              mse_thresh = evol_params['mse'],
                              atol = evol_params['atol'],
                              init = evol_params['init'],
                              rank = rank, size = size, comm = comm)

elap = (time.time()- start_time)
print(rank, "_", elap)

if evol_params['optim'] =='w':
    MAT = c_mat
elif evol_params['optim'] =='c':
    MAT = w_mat
final_res = hf.residuals_cw_corr(res.x, wc_params, targ_data, nodes, tract_mat,
                                 MAT, skip, wc_seed, evol_params['optim'], evol_params['heavi'])

#saving the result in the folder for the appropriate jobid
file_name = ("{}/rank{}_{}").format(job, rank, final_res)
np.save(file_name, np.array(res.x))

#%% Extra Stuff

# =============================================================================
# _Dt = wc_params['dt']
# _alpha = 10
# _dt = _Dt/_alpha # time step is 1ms: _dt = 0.001
# fs = 1/_dt # Sampling rate, or number of measurements per second
# =============================================================================
