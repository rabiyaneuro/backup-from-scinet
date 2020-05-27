# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:35:45 2018

@author: Administrator
"""
#%% Import modules

from helper_hpc import * 
import wc_model_sim_functions as wc
from differentialevolution_par import *

from mpi4py import MPI

#initialize the MPI interface
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#%% Seed

#initialize random num gen for each process
rng = np.random.RandomState(rank+1)

#Set seed for the wc_model_sim
wc_seed = 0

# Set seed for generating the target data
seed_targ = 2018
#%% Set constants

g = 0.095 #need low g for a bigger matrix to avoid overflow of exp
cond_vel = 4000
c1 = 1.6
c2 = -4.7
c3 = 3
c4 = -0.63
I_e = 1.8
I_i = -0.2
nodes = 76
time_steps = 1500
dt = 0.01
d = 0.001
#%% load cocomaq weights and tract length and setting target data

tract_mat = np.load("tract_mat.npy")
W_mat = np.load("W_mat.npy")

ue_targ_new, ui_targ_new = wc.wc_model_sim_new(g, cond_vel,
                                            c1, c2, c3, c4, I_e, I_i, nodes, tract_mat, W_mat, 
                                            dt = dt, time_steps = time_steps,  
                                            d = d, seed_num = seed_targ)

targ_data = plot_cor_mat(ue_targ_new, nodes, 200, False)
#%% 
def residuals(p, final = False):
    ind = 0
    wmat = np.zeros((nodes, nodes))
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            wmat[row,col] = p[ind]
            wmat[col,row] = p[ind]
            ind += 1
    ue_array, _ = wc.wc_model_sim_new(g, cond_vel, c1, c2, c3, c4, I_e, I_i, nodes, 
                                            tract_mat, wmat, dt = dt, time_steps = time_steps,
                                            d = d, seed_num = wc_seed)
    
    exp_data = plot_cor_mat(ue_array, nodes, 200)
    
    #quantity we are trying to minimize is the mse 
    res = mse(targ_data,exp_data)
    if final:
        ### CHange the lower corr bound to -1 if there are neg correlations ##
        plot_mat(targ_data, "targ corr", 1, 0)
        plot_mat(exp_data,"opt corr", 1, 0)
        print(mse(targ_data, exp_data))
        
        plot_mat(W_mat, "targ weight", 1, 0)
        plot_mat(wmat,"Opt weight", 1, 0)
        print(mse(W_mat, wmat))

        return res
    return res
#%%
t = time.time()
bounds = []
param_cnt = int((((nodes**2)-nodes)/2))
for n in range(param_cnt):
    bounds.append((0,1))
    

pop = 5
prior = []
for p in range(pop):
    prior.append(matrix2p(targ_data))

res= differential_evolution(residuals, strategy = 'best2bin', popsize = pop,
                            recombination = 0.9, bounds = bounds, seed = rng, disp = True, polish = False,
                            maxiter = 120, prior = prior, init = 'prior', rank = rank,
                            size = size, comm = comm)

elap = (time.time() - t)/60/60
print(elap)

np.save("rank"+str(rank)+"_"+str(elap), np.array(res.x)) 