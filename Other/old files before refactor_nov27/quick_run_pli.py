# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:35:45 2018
pli version for scinet

@author: Administrator
"""
#%% Import modules

from helper_hpc import * 
import wc_model_sim_functions as wc
from differentialevolution_par import *
import mne

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

#%% Set constants
g = 0.1 #need low g for a bigger matrix to avoid overflow of exp
cond_vel = 4000
c1 = 1.6
c2 = -4.7
c3 = 3
c4 = -0.63
I_e = 1.8
I_i = -0.2
time_steps = 12000
dt = 0.1 #is actually Dt and is used as the delta_t in my model when integrating
d = 0.001
epoch_size = 200
normalize = True
single_epoch = True
epoch_num = 1 #epochs go from 0-12

#signal properties
_Dt = dt
_alpha = 10
_dt = _Dt/_alpha # time step is 1ms: _dt = 0.001
fs = 1/_dt # Sampling rate, or number of measurements per second
#%% Target averaged PLI matrix from sick kids dataset

adjmat = np.load('adjmat.npy')
_, nodes, time_chunks, freq_bands = np.shape(adjmat)

if not single_epoch:
    #want to average over all time chunks in freq band : 13-29 (4th element in freq_bands)
    all_time = []
    for i in range(time_chunks):
        all_time.append(adjmat[:,:,i,3])
    targ_data = np.mean( np.array(all_time), axis=0 )
else:
    targ_data = adjmat[:,:,epoch_num,3]

if normalize:
    targ_data = targ_data/np.max(targ_data)

targ_data_p = matrix2p(targ_data)
tract_mat = np.zeros((nodes,nodes))
#%%
def plot_avg_pli(ts, len_chunk, timesteps, fmin= 17, fmax =20, norm = True):
    ue_targ_chunks= []
    num_chunks = int(timesteps/len_chunk)
    
    for ch in range(1,num_chunks):
        ue_targ_chunks.append(ts[:,ch*len_chunk:ch*len_chunk + len_chunk])
    
    ue_targ_chunks = np.array(ue_targ_chunks)
    
    _con, _freqs, _times, _n_epochs, _n_tapers = mne.connectivity.spectral_connectivity(
            data=ue_targ_chunks,method='pli', sfreq=fs, fmin=fmin,fmax=fmax, 
            mode = 'fourier', verbose = False)
    
    #average all
    all_ue_pli = []
    for i in range(num_chunks):
        all_ue_pli.append(_con[:,:,0]) 
        
    avg_ue_pli = np.mean( np.array(all_ue_pli), axis=0 )
    if norm:
        avg_ue_pli = avg_ue_pli/np.max(avg_ue_pli)

    return avg_ue_pli
#%%

def residuals_pli(p, final = False):
    ind = 0
    wmat = np.zeros((nodes, nodes))
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            wmat[row,col] = p[ind]
            wmat[col,row] = p[ind]
            ind += 1
    ue_array, _ = wc.wc_model_sim_new(g, cond_vel, 
                                            c1, c2, c3, c4, I_e, I_i, nodes, tract_mat, wmat,
                                            dt = dt, time_steps = time_steps, d = d, seed_num = wc_seed)
    exp_data = plot_avg_pli(ue_array,epoch_size, time_steps, normalize)
    exp_data_p = matrix2p(exp_data, upper = False) #lower triang matrix
    
    #quantity we are trying to minimize is the mse 
    res = mse(targ_data_p,exp_data_p)
    if final:
        ### CHange the lower corr bound to -1 if there are neg correlations ##
        plot_mat(targ_data, "targ pli")
        plot_mat(exp_data,"opt pli", half = True)
        print(mse(targ_data_p,exp_data_p))
        plot_mat(wmat,"Opt weight",1,0)
        return res
    return res
#%%
t = time.time()
bounds = []
param_cnt = int((((nodes**2)-nodes)/2))
for n in range(param_cnt):
    bounds.append((0,1))
    

pop = 5
#prior = []
#for p in range(pop):
#    prior.append(matrix2p(targ_data))

res= differential_evolution(residuals_pli, strategy = 'best2bin', popsize = pop,
                            recombination = 0.9, bounds = bounds, 
                            seed = rng, disp = False, polish = False,
                            maxiter = 2, rank = rank,
                            size = size, comm = comm)

elap = (time.time() - t)/60/60
print(elap)
final_res = str(residuals_pli(res.x))
np.save(final_res+"_rank"+str(rank), np.array(res.x))
