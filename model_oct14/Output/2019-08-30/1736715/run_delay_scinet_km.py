# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:58:53 2019

@author: Rabiya Noori
"""

# -*- coding: utf-8 -*-
"""
Created on Aug 10

This is for running on SciNet
This file sets parameters for the KM brain scale model-  It runs the
optimization to find the best delays to generate the system which best matches the target

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

res = df.differential_evolution(hf.residuals_km, args = args,
                              strategy = evol_params['strategy1'],
                              popsize = evol_params['popsize'],
                              mutation = evol_params['mut'],
                              recombination = evol_params['recomb'],
                              bounds = evol_params['bounds'],
                              tol = evol_params['tol'],
                              maxiter = evol_params['maxiter'],
                              seed = rng,
                              disp = False,
                              polish = evol_params['polish'],
                              mse_thresh = evol_params['mse'],
                              atol = evol_params['atol'],
                              init = evol_params['init'],
                              thresh = 0.0001,
                              strategy1 = evol_params['strategy1'],
                              strategy2 = evol_params['strategy2'],
                              rank = rank, size = size, comm = comm,
                              prior= evol_params['prior'], jobid = job)

elap = (time.time()- start_time)
print(rank, "_", elap)

vector_est = res[0].x
vector_costs = np.array(res[1])
vector_best_sols = np.array(res[2])

final_res = hf.residuals_km(vector_est, km_params, nodes, targ_data, w_mat, 
                            upper_b, evol_params['res_metric'])

np.save('{}/vector_est_rank{}_cost{}.npy'.format(job, rank, final_res), vector_est)
np.save('{}/vector_costs_rank{}_cost{}.npy'.format(job, rank, final_res), vector_costs)
np.save('{}/vector_best_sols_rank{}_cost{}.npy'.format(job, rank, final_res), vector_best_sols)


