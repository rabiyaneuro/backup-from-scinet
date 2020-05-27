# -*- coding: utf-8 -*-
"""
Created on March 20

This is for running on SciNet
It runs the
optimization to find the best delays to generate the system which has a PLI+Corr
that best matches the target


Feb 8 - This is the same as ren_delay_scinet but uses a diff modelling function in 
residuals so it accepts c_mat AND a w_mat (for use with generated data)

- so far only works with optimizing conduction vel

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
"""residuals_cw_corr(c, params, nodes, targ_data, cw_mat, skip, seed, optim, tract_mat=None, heavyside =False,
                   plot = None):"""

start_time = time.time()

res = df.differential_evolution(hf.residuals_pli_corr, args = args,
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
                              rank = rank, size = size, comm = comm,
                              prior= evol_params['prior'], jobid = job)

elap = (time.time()- start_time)
print(rank, "_", elap)

final_res = res.fun


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
