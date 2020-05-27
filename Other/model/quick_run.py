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

g = 0.01 #need low g for a bigger matrix to avoid overflow of exp
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

targ_data = plot_cor_mat(ue_targ_new, nodes, 400, False)
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
    
    exp_data = plot_cor_mat(ue_array, nodes, 400)
    
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
                            recombination = 0.9, bounds = bounds, 
                            tol = 0.5 , seed = rng, disp = True, polish = False,
                            mse_thresh = 0.1, maxiter = 40, prior = prior, init = 'prior', rank = rank,
                            size = size, comm = comm)

elap = (time.time() - t)/60/60
print(elap)

np.save("rank"+str(rank)+"_"+str(elap), np.array(res.x)) 

#%%
#def weights(nodes):
#    np.random.seed(2018) 
#    
#    W_mat = np.ones((nodes, nodes))
#    np.fill_diagonal(W_mat, 0)
#    
#    # changing weights from 1 to random number 
#    for row in range(0,nodes):
#        for col in range(row+1, nodes):
#            W_mat[row,col] = W_mat[row, col] * np.random.randint(0,100)/100
#            W_mat[col,row] = W_mat[row,col]
#    return W_mat
#W_mat = weights(nodes)
#
#def tracts(nodes):
#    np.random.seed(2018) 
#    tract_mat = np.ones((nodes, nodes))
#    np.fill_diagonal(tract_mat, 0)
#    
#    # changing weights from 1 to random number 
#    for row in range(0,nodes):
#        for col in range(row+1, nodes):
#            tract_mat[row,col] = tract_mat[row, col] * np.random.randint(0,100)/100
#            tract_mat[col,row] = tract_mat[row,col]
#            
#    return tract_mat
#tract_mat = tracts(nodes)

#np.random.seed(2018) 
#ue_targ_new, ui_targ_new = wc.wc_model_sim_new(g, cond_vel, tract_mat, W_mat,
#                                            c1, c2, c3, c4, I_e, I_i, nodes, 
#                                            time_steps = time_steps, dt = dt, d = d)
#

##
#targ_data = plot_cor_mat(ue_targ_new, nodes, False)

#%%
"""
: array([0.95920395, 0.67426886, 0.907738  , ..., 0.04404392, 0.75639681,
       0.03586106])
    
    
    array([0.83501648, 0.19191432, 0.64877838, ..., 0.86247454, 0.77293313,
       0.81313302]
"""

#prior = [matrix2p(W_mat)]*100






#%%
#popsize = 15
#mut = 0.6
#recomb = 0.6
#tol = 0.5
#atol = 0
#maxiter = 50
#seed = 180
#res= differential_evolution(residuals, strategy = 'best2bin', popsize = 40, 
#                                     recombination = 0.9, bounds = bounds, 
#                                     seed = SEED, disp = True, 
#                                     polish = True, mse_thresh = 0.1)
#elap = (time.time() - t)/60/60
#print(elap)
#np.save("res"+str(elap), np.array(res.x)) 


#residuals(res.x, True)
#SEED = 2018
#t = time.time()
#bounds = []
#param_cnt = int((((nodes**2)-nodes)/2))
#for n in range(param_cnt):
#    bounds.append((0,1))
#    
#popsize = 15
#mut = 0.6
#recomb = 0.6
#tol = 0.5
#atol = 0
#maxiter = 50
#seed = 180
#res= differential_evolution(residuals, strategy = 'best2bin', popsize = 40, 
#                                     recombination = 0.9, bounds = bounds, 
#                                     seed = SEED, disp = True, 
#                                     polish = True, mse_thresh = 0.1)
#elap = (time.time() - t)/60/60
#print(elap)
#np.save("res"+str(elap), np.array(res.x)) 

#residuals(res.x, True)

#iter 4- f(x) ~0.58
#6 -0.586083
#same for 10,22
#iter 137, 138, 181, 294:  0.569912

#1000: 0.557696
#%%
# differential_evolution step 78: f(x)= 0.547059
# same for iter 79, 83, 85

#differential_evolution step 123: f(x)= 0.538439 - accidentaly stopped it by
#pressing ctrl+c when i tried to copy something, the elap time was 66 hours
# so around 2.75 days

#SEED = 90
#t = time.time()
#bounds = []
#param_cnt = int((((nodes**2)-nodes)/2))
#for n in range(param_cnt):
#    bounds.append((0,1))
#    
#popsize = 15
#mut = 0.6
#recomb = 0.6
#tol = 0.5
#atol = 0
#maxiter = 50
#
#res76 = differential_evolution(residuals, strategy = 'rand1bin', popsize = 712, 
#                                     mutation = mut, recombination = recomb, bounds = bounds, 
#                                     tol = tol , seed = SEED, disp = True, 
#                                     polish = True, prior = prior, init = 'prior', mse_thresh = 0.1)
#elap = (time.time() - t)/60/60
#print(elap)
#
#residuals(res76.x, True)


#testing effects of seed


#rng = np.random.RandomState(2018)
#for i in range(5):
#    print(rng.randint(low = 0, high = 10, size = 1))
#    
#    
#np.random.seed(2018)
#for i in range(5):
#    np.random.seed(2018)
#    print(np.random.randint(low = 0, high = 10, size = 1))