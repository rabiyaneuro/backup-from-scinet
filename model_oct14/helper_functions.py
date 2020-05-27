"""
This file contains all the helper functions used in the simulations for the runs
on SciNet and locally

@author: Rabiya N
"""
#%% Import Statements

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
import random
import sklearn.preprocessing as pr
from scipy import signal
#from scipy.optimize import differential_evolution
from numpy import linalg as LA
import numexpr as ne
#from math import exp
import time
import mne

import wc_model_sim_functions as wcm
#%%% Cost functions
def residuals_c_corr(c, params, targ_data, nodes, tract_mat, skip, seed,
                   final = False):
    """ This function is used to optimize the network model based on the
    correlation matrix
    
    c: vector that you are trying to optimize - e.g. conduction velocity matrix
    converted to a vector, ranges from 0-1700 (0-17m/s)
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """
    ind = 0
    cmat = np.ones((nodes, nodes))
    
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            cmat[row,col] = c[ind]
            cmat[col,row] = c[ind]
            ind += 1
    cmat = np.reciprocal(cmat, where = cmat>0)
#    print(cmat, "CMAAT")
    ue_array, _, delays = wcm.wc_modelsim_c(params, tract_mat, cmat, nodes,
                                       seed_num = seed)
    
    exp_data = plot_cor_mat(ue_array, nodes, skip)
    
    
    #quantity we are trying to minimize is the mse 
    res = mse(np.triu(targ_data,1).ravel(),np.triu(exp_data,1).ravel())
    
    if final:
        np.fill_diagonal(exp_data, 0)
        plot_mat(targ_data, "targ corr", 1, -1)
        plot_mat(exp_data,"opt corr", 1, -1)
        print("MSE", res)
        plot_mat(delays,"Delays")
        return res
    return res
#%%
def residuals_cw_corr_sparse(c, params, nodes, targ_data, cw_mat, skip, seed, optim, tract_mat,
                             heavyside =False, plot = None, vmax_ = None, vmin_ = None):
    
    """ This function is used to optimize the network model based on the
    correlation matrix of generated data for a sparse matrix
    
    before using this, make sure wmat and tractmat and 0 in the same places
    
    You can use this to either estimate the conduc velocity or weights distribution
    
    c: conduction velocity matrix converted to a vector
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """    
    
    if optim =="d":
        wmat = cw_mat
        wmat = np.zeros((nodes, nodes))
        delays_mat = (p2matrix(np.round(c),nodes)).astype(int)
       
        ue_array, _, delays = wcm.wc_model_sim_d(params, delays_mat, wmat, nodes, seed_num= seed)
        exp_data = plot_cor_mat(ue_array, nodes, skip)

    else:
        if optim =="c":
            tract_p = matrix2p(tract_mat)
            wmat = cw_mat
            ind = 0
            cind = 0
            cmat = np.ones((nodes, nodes))
            
            for row in range(0,nodes):
                for col in range(row+1, nodes):
                    if tract_p[ind] > 0:
                        cmat[row,col] = c[cind]
                        cmat[col,row] = c[cind]
                        cind +=1
                    else:
                        cmat[row,col] = 0
                        cmat[col,row] = 0
                    ind += 1
            if type(cmat[0,0]) != type(1.0):
                cmat =cmat*1.0
            cmat_r = np.reciprocal(cmat, where= cmat>0) #does not work with integars
            np.fill_diagonal(cmat_r,0)
            
        elif optim == "w":
            cmat_r = cw_mat
            ind = 0
            wmat = np.ones((nodes, nodes))
            
            for row in range(0,nodes):
                for col in range(row+1, nodes):
                    if heavyside:
                        val = np.heaviside(c[ind],0.5)
                    else:
                        val = c[ind]
                    wmat[row,col] = val
                    wmat[col,row] = val
                    ind += 1
            np.fill_diagonal(wmat,0)
        
        #print("here")
        ue_array, _, delays = wcm.wc_model_sim_new(params, tract_mat, cmat_r, wmat, 
                                                   nodes, seed_num = seed)    
#        if plot == "c":
#            plot_cor_mat(ue_array, nodes, skip, True)
#            plt.figure(1, figsize=(15,6))
#            for sing_node in range(nodes):
#                plt.plot(ue_array[sing_node], '-k', linewidth=0.2)
#            avg = np.mean(ue_array, axis= 0 )
#            plt.plot(avg, '-k', linewidth = 0.8)
#            plt.xlabel("Time (ms)", fontsize = "xx-large")
#            plt.ylabel("$u_e$", fontsize = "xx-large")
#            plt.xticks(fontsize = 20)
#            plt.yticks(fontsize = 20)
#            plt.show()
        exp_data = plot_cor_mat(ue_array, nodes, skip)
    

    #quantity we are trying to minimize is the mse 
    res = mse(np.triu(targ_data,1).ravel(),np.triu(exp_data,1).ravel())    
    
    if plot == "c":
        np.fill_diagonal(cmat,0)
        return(cmat/1000, ue_array, res, vmax_ , vmin_)
    return res
#%%
def residuals_cw_corr(c, params, nodes, targ_data, cw_mat, skip, seed, optim, bound, tract_mat=None, heavyside =False,
                   plot = None, vmax_ = None, vmin_ = None):
    
    """ This function is used to optimize the network model based on the
    correlation matrix of generated data
    
    You can use this to either estimate the conduc velocity or weights distribution
    
    c: conduction velocity matrix converted to a vector
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """    
    if optim =="d":
        wmat = cw_mat
        
        delays_mat = (p2matrix(c,nodes)).astype(int)
        
        #print("in residuals fxn with delay mat - ", delays_mat) -- confirmed converting to matrix correctly
       
        ue_array, _, delays = wcm.wc_model_sim_d(params, delays_mat, bound, wmat, nodes, seed_num= seed)
        
        exp_data = plot_cor_mat(ue_array, nodes, skip)

#        plt.figure(3, figsize=(35,6))
##        for sing_node in range(nodes):
##            plt.plot(ue_array[sing_node], '-k', linewidth=0.2)
#        avg = np.mean(ue_array, axis= 0 )
#        plt.plot(avg, '-k', linewidth = 0.8)
#        plt.xlabel("Time (ms)", fontsize = "xx-large")
#        plt.ylabel("$u_e$", fontsize = "xx-large")
#        plt.xticks(fontsize = 20)
#        plt.yticks(fontsize = 20)
#        plt.show()

    else:
        if optim =="c":
            wmat = cw_mat
            ind = 0
            cmat = np.ones((nodes, nodes))
            
            for row in range(0,nodes):
                for col in range(row+1, nodes):
                    cmat[row,col] = c[ind]
                    cmat[col,row] = c[ind]
                    ind += 1
            cmat_old = np.ndarray.copy(cmat)

            cmat = np.reciprocal(cmat)
            np.fill_diagonal(cmat,0)

            
        elif optim == "w":
            cmat = cw_mat
            ind = 0
            wmat = np.ones((nodes, nodes))
            
            for row in range(0,nodes):
                for col in range(row+1, nodes):
                    if heavyside:
                        val = np.heaviside(c[ind],0.5)
                    else:
                        val = c[ind]
                    wmat[row,col] = val
                    wmat[col,row] = val
                    ind += 1
            np.fill_diagonal(wmat,0)
        
        print("wrong place for est delays")
        ue_array, _, delays = wcm.wc_model_sim_new(params, tract_mat, cmat, wmat, 
                                                   nodes, seed_num = seed, lowest_c = bound)    
        exp_data = plot_cor_mat(ue_array, nodes, skip)
    
    
    #quantity we are trying to minimize is the mse 
    res = mse(np.triu(targ_data,1).ravel(),np.triu(exp_data,1).ravel())
    
    #plot_mat(targ_data, "targ corr", 1, -1)
    
    if plot == "cw":
        np.fill_diagonal(cmat_old,0)
        return(cmat_old/1000, ue_array, res, vmax_ , vmin_)
        
#        import seaborn as sns
#        skip = 200
#        np.fill_diagonal(exp_data, 0)
#        plt.figure()
#        sns.heatmap(exp_data, cmap ="viridis", vmax = 1, vmin =-1)
#        plt.xlabel("Node")
#        plt.ylabel("Node")
#        plt.show()
#        print("MSE", res)
#        
#        plt.figure(3, figsize=(15,6))
#        for sing_node in range(nodes):
#            plt.plot(ue_array[sing_node], '-k', linewidth=0.2)
#        avg = np.mean(ue_array, axis= 0 )
#        plt.plot(avg, '-k', linewidth = 0.8)
#        plt.xlabel("Time (ms)", fontsize = "xx-large")
#        plt.ylabel("$u_e$", fontsize = "xx-large")
#        plt.xticks(fontsize = 20)
#        plt.yticks(fontsize = 20)
#        plt.show()
#        
#        print("cmap")
#        plt.figure()
#        cmat_old = cmat_old/1000
#        np.fill_diagonal(cmat_old, 0)
#        sns.heatmap(cmat_old, vmax = vmax_, vmin = vmin_, cmap = "viridis")
#        plt.show()
#        
#        if optim =="w":
#            plot_mat(wmat, "opt w", 0, 1)
            
    if plot == "delay":
        plot_mat(delays,"Delays")

    return res
#%%
def residuals_corr(p, params, targ_data, nodes, tract_mat, skip, seed,
                   final = False, W_mat = None):
    """ This function is used to optimize the network model based on the
    correlation matrix
    
    p: vector that you are trying to optimize - e.g. weights matrix converted
    to a vector
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """

    ind = 0
    wmat = np.zeros((nodes, nodes))
    
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            wmat[row,col] = p[ind]
            wmat[col,row] = p[ind]
            ind += 1

    ue_array, _ = wcm.wc_model_sim_new(params, tract_mat, wmat, nodes,
                                       seed_num = seed)
    
    exp_data = plot_cor_mat(ue_array, nodes, skip)
    
    
    #quantity we are trying to minimize is the mse 
#    res = mse(targ_data,exp_data)
    res = mse(np.triu(targ_data,1).ravel(),np.triu(exp_data,1).ravel())
    
    if final:
        np.fill_diagonal(exp_data, 0)
        ### CHange the lower corr bound to -1 if there are neg correlations ##
        plot_mat(targ_data, "targ corr", 1, -1)
        plot_mat(exp_data,"opt corr", 1, -1)
        print("MSE", res)
        if not W_mat is None:
            plot_mat(W_mat, "targ weight", 1, 0)
            print(mse(W_mat, wmat))
        plot_mat(wmat,"Opt weight", 1, 0)
        return res
    return res

#%%
def residuals_pli(c, params, fs, nodes, targ_data, w_mat, time_chunk, skip, seed, optim, 
                  tract_mat, heavi, fmin, fmax, plot= False):
    """
     wc_params, fs, nodes, targ_data, MAT, 
                                     skip, wc_seed, evol_params['optim'], tract_mat, 
                                     heavyside= evol_params['heavi']
    This function is used to optimize the network model based on the
    PLI matrix
    
    c: vector that you are trying to optimize - e.g. weights matrix converted
    to a vector- RIGHT NOW IT ONLY WORKS WITH CONDUCTION VELOCITY
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """
    
    ind = 0
    cmat = np.ones((nodes, nodes))
    
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            cmat[row,col] = c[ind]
            cmat[col,row] = c[ind]
            ind += 1
    if type(cmat[0,0]) != type(1.0):
        cmat =cmat*1.0
    cmat_old = np.ndarray.copy(cmat)
    cmat = np.reciprocal(cmat, where= cmat>0.0)
    np.fill_diagonal(cmat,0)

    ue_array, _, delays = wcm.wc_model_sim_new(params, tract_mat, cmat, w_mat, 
                                               nodes, seed_num = seed)    
   
    
    exp_data = plot_avg_pli(ue_array,time_chunk, params['time_steps'], skip, fs, fmin, fmax)
    
    exp_data_p = matrix2p(exp_data, upper = False)
    targ_data_p = matrix2p(targ_data, upper = False)#lower triang matrix
    
    #quantity we are trying to minimize is the mse 
    res = mse(targ_data_p,exp_data_p)
    if plot:
        np.fill_diagonal(cmat_old,0)
        return(cmat_old/1000, ue_array, res, exp_data)
        
    return res

#%%
def residuals_pli_corr(c, params, fs, nodes, targ_data_corr, targ_data_pli, 
                       w_mat, chunk, skip, seed, optim, 
                       tract_mat, heavi, fmin, fmax, plot = False):          
                       
    """
    This function is used to optimize the network model based on the
    PLI & Correlation matrix
    
    c: vector that you are trying to optimize - e.g. weights matrix converted
    to a vector- RIGHT NOW IT ONLY WORKS WITH CONDUCTION VELOCITY
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """
    
    ind = 0
    cmat = np.ones((nodes, nodes))
    
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            cmat[row,col] = c[ind]
            cmat[col,row] = c[ind]
            ind += 1
    if type(cmat[0,0]) != type(1.0):
        cmat =cmat*1.0
    cmat_old = np.ndarray.copy(cmat)
    cmat = np.reciprocal(cmat, where= cmat>0.0)
    np.fill_diagonal(cmat,0)

    ue_array, _, delays = wcm.wc_model_sim_new(params, tract_mat, cmat, w_mat, 
                                               nodes, seed_num = seed)    
   
    #PLI
    exp_data_pli = plot_avg_pli(ue_array,chunk, params['time_steps'], skip, fs, fmin, fmax)
    
    exp_data_p = matrix2p(exp_data_pli, upper = False)
    targ_data_p = matrix2p(targ_data_pli, upper = False)#lower triang matrix
    
    #quantity we are trying to minimize is the mse 
    res_pli = mse(targ_data_p,exp_data_p)
    
    
    #correlation
    exp_data_corr = plot_cor_mat(ue_array, nodes, skip)
    res_corr = mse(np.triu(targ_data_corr,1).ravel(),np.triu(exp_data_corr,1).ravel())
    
    final_res = res_corr+ res_pli
    if plot:
        np.fill_diagonal(cmat_old,0)
        return(cmat_old/1000, ue_array, final_res, exp_data_pli, exp_data_corr)
        
    return final_res

#%%
def residuals_mne_delay(d, params, fs, nodes, targ_data, 
                       w_mat, chunk, skip, seed, 
                       metric, upper, fmin, fmax, plot = False):          
                       
    """
    This function is used to optimize the network model based on metrics from
    mne library
    
    d: vector that you are trying to optimize - i.e. delay matrix converted
    to a vector
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains c1, c2, c3, c4, I_e, I_i, dt, time_steps, d
    """
    
    ind = 0
    dmat = np.ones((nodes, nodes))
    
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            dmat[row,col] = d[ind]
            dmat[col,row] = d[ind]
            ind += 1
    np.fill_diagonal(dmat,0)
    dmat = dmat.astype(int)

    ue_array, _, _ = wcm.wc_model_sim_d(params, dmat, upper, w_mat, nodes,
                                       seed_num = seed) 
   
    #mne metric
    exp_data = plot_avg_mne(ue_array,chunk, params['time_steps'], skip, fs, 
                            fmin, fmax, metric)
    
    exp_data_p = matrix2p(exp_data, upper = False)
    targ_data_p = matrix2p(targ_data, upper = False)#lower triang matrix
    
    #quantity we are trying to minimize is the mse 
    res_pli = mse(targ_data_p,exp_data_p)
    
    return res_pli

#%%
    
def residuals_km(d, params, nodes, targ_data, w_mat, upper, method = "r_avg"):          
                       
    """
    This function is used to optimize the network model based on kurumoto
    model's order parameter
    
    d: vector that you are trying to optimize - i.e. delay matrix converted
    to a vector
    
    params: dict containing the wc and simulation params for the population
    models at each node; contains 
    """
    
    ind = 0
    dmat = np.ones((nodes, nodes))
    
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            dmat[row,col] = d[ind]
            dmat[col,row] = d[ind]
            ind += 1
    np.fill_diagonal(dmat,0)
    dmat = dmat.astype(int)

    ue_array = wcm.km_model_sim_d(params, dmat, upper, w_mat, nodes)
    
   
    #order param
   
    if method == "r_avg":
        exp_data = order_param_matrix(ue_array, nodes, skip= upper+500)
        
        exp_data_p = matrix2p(exp_data)
        targ_data_p = matrix2p(targ_data)
        
        #quantity we are trying to minimize is the mse 
        res_km = mse(targ_data_p,exp_data_p)
        
        return res_km
    
    elif method == "time":
        #take the order param between teh series and commare to 1
        order_params = order_param_time(ue_array, targ_data, skip= upper+500)
        res_km = mse(order_params, np.ones_like(order_params))
        
        #tempp
#        for sing_node in range(nodes):
#            plt.plot(ue_array[sing_node], '-k', linewidth=0.2)
#        plt.xlabel("Time (ms)", fontsize = "xx-large")
#        plt.ylabel("$theta$", fontsize = "xx-large")
#        plt.xticks(fontsize = 20)
#        plt.yticks(fontsize = 20)
#        plt.show()

        
        return res_km
    elif method == "mse":
        #mse between time series
        # skip first 500 + upper timesteps
        res_km = mse(targ_data[:,upper+500:],ue_array[:,upper+500:])
        return res_km
#%%
#def residuals_pli_old(p, final = False):
#
#    ind = 0
#    wmat = np.zeros((nodes, nodes))
#    for row in range(0,nodes):
#        for col in range(row+1, nodes):
#            wmat[row,col] = p[ind]
#            wmat[col,row] = p[ind]
#            ind += 1
#    ue_array, _ = wc.wc_model_sim_new(g, cond_vel, 
#                                            c1, c2, c3, c4, I_e, I_i, nodes, tract_mat, wmat,
#                                            dt = dt, time_steps = time_steps, d = d, seed_num = seed_)
#    exp_data = plot_avg_pli(ue_array,300, time_steps)
#    exp_data_p = matrix2p(exp_data, False) #lower triang matrix
#    
#    #quantity we are trying to minimize is the mse 
#    res = mse(targ_data_p,exp_data_p)
#    if final:
#        ### CHange the lower corr bound to -1 if there are neg correlations ##
#        upper = np.max((np.max(targ_data_p), np.max(exp_data_p)))
#        plot_mat(targ_data, "targ corr")
#        plot_mat(exp_data,"opt corr", _max = upper)
#        print(mse(targ_data_p,exp_data_p))
#        
#        plot_mat(wmat,"Opt weight",1,0)
#
#        return res
#    return res
#%% kuramoto stuff
        
def order_param(node_i, node_j,skip):
    """
    calculates the order param between two KM oscillators and averages over 
    time to return one value
    
    # np.absolute takes the modulus of real+imaginary number
    """
    sum_phases = 0
    for i in range(np.size(node_i)):
        sum_phases = sum_phases+ np.absolute(np.exp(1j*node_i[i])+
                                             np.exp(1j*node_j[i]))
    r = (sum_phases)/np.size(node_i)/2
    return r

def order_param_time(data_1, data_2,skip):
    """
    calculates the order param between two KM oscillators and returns the 
    vector with elements for each time point
    # np.absolute takes the modulus of real+imaginary number

    # sometimes even if you input the same array for data1 and data2, the r
    will not be 1 because of precision/rounding error
    """
    if data_1.shape != data_2.shape:
        return False
    
    matrix_r = []
    
    for node in range(data_1.shape[0]):
        r = []
        for i in range(np.size(data_1[node])):
            r.append(np.absolute(np.exp(1j*data_1[node][i])+
                                             np.exp(1j*data_2[node][i]))/2)
        matrix_r.append(r)
    
    return np.array(matrix_r)

def order_param_matrix(theta_array, nodes, skip):
    r_matrix = np.ones((nodes,nodes))
    for i in range(nodes):
        for j in range(nodes):
            r_matrix[i,j] = order_param(theta_array[i],theta_array[j], skip )
    return r_matrix
#%%
def plot_avg_pli(ts,time_chunk, timesteps,skip, fs, f_min, f_max ):
    ue_targ_chunks= []
    num_chunks = int((timesteps-skip)/time_chunk)
    
    for ch in range(num_chunks):
        ue_targ_chunks.append(ts[:,skip:][:,ch*time_chunk:ch*time_chunk + time_chunk])
    
    
    _con, _freqs, _times, _n_epochs, _n_tapers = mne.connectivity.spectral_connectivity(
        data=ue_targ_chunks,method='wpli', fmin =f_min, fmax =f_max, sfreq=fs, faverage = True,
        mode = 'fourier', verbose= False)
        
    return _con[:,:,0]

def plot_avg_mne(ts,time_chunk, timesteps,skip, fs, f_min, f_max, method):
    ue_targ_chunks= []
    num_chunks = int((timesteps-skip)/time_chunk)
    
    for ch in range(num_chunks):
        ue_targ_chunks.append(ts[:,skip:][:,ch*time_chunk:ch*time_chunk + time_chunk])
    
    
    _con, _freqs, _times, _n_epochs, _n_tapers = mne.connectivity.spectral_connectivity(
        data=ue_targ_chunks,method=method, fmin =f_min, fmax =f_max, sfreq=fs, faverage = True,
        mode = 'fourier', verbose= False)
        
    return _con[:,:,0]
#%% GENERATING DATA
    
def weights(nodes, seed):
    np.random.seed(seed) 
    
    W_mat = np.ones((nodes, nodes))
    np.fill_diagonal(W_mat, 0)
    
    # changing weights from 1 to random number 
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            W_mat[row,col] = W_mat[row, col] * np.random.randint(0,100)/100
            W_mat[col,row] = W_mat[row,col]
    return W_mat

def tracts(nodes, seed):
    np.random.seed(seed) 
    tract_mat = np.ones((nodes, nodes))
    np.fill_diagonal(tract_mat, 0)
    
    # changing weights from 1 to random number 
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            tract_mat[row,col] = tract_mat[row, col] * np.random.randint(0,100)/100
            tract_mat[col,row] = tract_mat[row,col]
            
    return tract_mat

def anat_data(nodes, seed, lmin, lmax):
    np.random.seed(seed) 
    
    mat = np.ones((nodes, nodes))
    np.fill_diagonal(mat, 0)
    
    # changing weights from 1 to random number 
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            mat[row,col] = mat[row, col] * np.random.randint(lmin,lmax*100)/100
            mat[col,row] = mat[row,col]
    return mat


#%% Frequency, in cycles per second, or Hertz
# source: https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html

#from scipy import fftpack
#
#X = fftpack.fft(ue_targ_new[0])
#freqs = fftpack.fftfreq(len(ue_targ_new[0])) * fs
#
#fig, ax = plt.subplots()
#
#ax.stem(freqs, np.abs(X))
#ax.set_xlabel('Frequency in Hertz [Hz]')
#ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
##ax.set_xlim(-fs / 4, fs / 4)
#ax.set_xlim(-25, 25)
#
#ax.set_ylim(-5, 110)
#

#%%timing

def time2run(fxn, *args):
    t = time.time()
    fxn(*args)
    elap1 = time.time() - t
    print(elap1)
    
#%%
def p2matrix(p, nodes):
    """fills in upper and lower triangle of matrix with p, leaving the diagonal with 0s"""
    ind = 0
    wmat = np.zeros((nodes, nodes))
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            wmat[row,col] = p[ind]
            wmat[col,row] = p[ind]
            ind += 1
    return wmat

def matrix2p(mat, upper = True):
    p = []
    for i in range(int(((np.shape(mat)[0]**2)-np.shape(mat)[0])/2)):
        p.append(0)
    ind = 0
    if upper:
        for row in range(0,np.shape(mat)[0]):
            for col in range(row+1, np.shape(mat)[0]):
                 p[ind]= mat[row,col]
                 ind += 1        
    else:
        for row in range(1,np.shape(mat)[0]):
            for col in range(0, row):
                 p[ind]= mat[row,col]
                 ind += 1   
        
    return p
#%% math
    
#used in wc_sim_node
def sig(u):
     return (1/(1 + np.exp(-50*u)))
 
#used in wc_sim_node_new
def sig_mat(u):
     return ne.evaluate("1/(1 + exp(-50*u))")

def sig_mat_np(u):
    return 1/(1 + np.exp(-50*u))
 
def tau(t,c, timestep):
    '''
    t: tract length matrix
    c: conduction speed
    timestep: timestep used to solve ode
    
    returns an int representing the lag in # of timesteps
    '''
    return (np.round((t/c)/timestep)).astype(int)

#not as efficient, but floating points seems same as with mult()
def mult_old(ar1, ar2):
    assert np.shape(ar1) == np.shape(ar2), "not same dim"
    res = 0
    size = np.shape(ar1)[0]
    for i in range(size):
        res = res + ar1[i]*ar2[i]
    # you could also do res = np.sum(ar1*ar2) and it would give 
    # diff result from mult() bc numpy has diff precision error
    return res

def mult(ar1, ar2):
    res = ne.evaluate("sum(ar1*ar2,1)")    
    return res

def mult_np(ar1, ar2):
    res = np.sum(ar1*ar2,1)    
    return res

#haven't tested the mse functions a second time for precision yet
def mse_old(A,B):
    return ((A - B) ** 2).mean(axis=None)

def mse(A,B):
    return ne.evaluate("(A - B) ** 2").mean(axis=None)

def mse_p(A,B):
    return ne.evaluate("(A - B) ** 2").mean(axis=None)

#%% sim

#def wc_sim_node(ce, ci, I, u_e, u_i, u, d, connec_strength, weights_mat,
#                delays_mat):
#    '''
#    u is either u_e or u_i
#    ce and ci are c1,c2,c3 or c4
#    I is either I_e or I_i
#    
#    fix the rng thing
#    '''
#    
#    base = -u + ce*sig(u_e) + ci*sig(u_i) + I
##    np.random.seed(SEED) 
##    print(np.random.get_state()[1][0])
#    noise = np.sqrt(2*d)*RNG.normal(0, 1, size=1)[0]
#
#    if (connec_strength!= 0):
#        connec = connec_strength*(mult_old(weights_mat, sig_mat(np.array(delays_mat))))
#        return base + noise + connec
#    
#    return base + noise
def wc_sim_node_no_w(ce, ci, I, u_e, u_i, u, d, connec_strength,
                    other_nodes, seed):
    '''
    u is either u_e or u_i
    ce and ci are c1,c2,c3 or c4
    I is either I_e or I_i
    
    other_nodes is a nxn matrix
    '''
    a = np.add(-u, np.multiply(ce,sig_mat(u_e)))
    b = np.add(np.multiply(ci,sig_mat(u_i)), I)
    base = np.add(a,b)

    noise = np.sqrt(2*d)*seed.normal(0, 1)
    
    connec = connec_strength*np.sum(sig_mat(np.array(other_nodes)),axis = 1)
    return base + noise + connec
    
def wc_sim_node_new(ce, ci, I, u_e, u_i, u, d, connec_strength, weights_mat,
                    delays_mat, seed):
    '''
    u is either u_e or u_i
    ce and ci are c1,c2,c3 or c4
    I is either I_e or I_i
    '''
    a = np.add(-u, np.multiply(ce,sig_mat(u_e)))
    b = np.add(np.multiply(ci,sig_mat(u_i)), I)
    base = np.add(a,b)
#    np.random.seed(2018) 
#    print(np.random.get_state()[1][0])

    noise = np.sqrt(2*d)*seed.normal(0, 1)
    
    if (connec_strength!= 0):        
        connec = connec_strength*(mult(weights_mat, sig_mat(np.array(delays_mat))))
        return base + noise + connec
    return base + noise

#def wc_sim_node_numexpr(ce, ci, I, u_e, u_i, u, d, connec_strength, weights_mat, delays_mat):
#    '''
#    u is either u_e or u_i
#    ce and ci are c1,c2,c3 or c4
#    I is either I_e or I_i
#    '''
#    sig_ue = sig_mat(u_e)
#    sig_ui = sig_mat(u_i)
#    
#    mult1 = ne.evaluate("ce*sig_ue")
#    a = ne.evaluate("-u + mult1")
#    
#    mult2 = ne.evaluate("ci*sig_ui")
#    b = ne.evaluate("mult2 + I ")
#    
#    base = ne.evaluate("a+b")
##    np.random.seed(2018) 
##    print(np.random.get_state()[1][0])
#
#    noise = np.sqrt(2*d)*RNG.normal(0, 1)
#    
#    if (connec_strength!= 0):        
#        connec = connec_strength*(mult(weights_mat, sig_mat(np.array(delays_mat))))
#        return base + noise + connec
#    return base + noise
#
#from numba import jit
#@jit(nopython=True)
#def wc_sim_node_numba(ce, ci, I, u_e, u_i, u, d, connec_strength, weights_mat, delays_mat):
#    '''
#    u is either u_e or u_i
#    ce and ci are c1,c2,c3 or c4
#    I is either I_e or I_i
#    '''
#    
#    a = np.add(-u, np.multiply(ce,1/(1 + np.exp(-50*u_e))))
#    b = np.add(np.multiply(ci,1/(1 + np.exp(-50*u_i))), I)
#    base = np.add(a,b)
##    np.random.seed(2018) 
##    print(np.random.get_state()[1][0])
#
#    noise = np.sqrt(2*d)*RNG.normal(0, 1)
#    
#    if (connec_strength!= 0):        
##        connec = connec_strength*(mult_np(weights_mat, sig_mat(np.array(delays_mat))))
#        connec = np.sum(weights_mat*np.array(delays_mat),1)   
#        return base + noise + connec
#    return base + noise


#%% plotting

def plot_single_node_figs(dt = 0.1, d = 0.001, time = [0]):
    '''
    plot u_e vs t, and u_i vs t  
    
    '''
    u_e = [0.1] 
    u_i = [0.1] 
    time_steps = 500
    for t in range(time_steps):
        u_e.append(u_e[t] + dt*wc_sim_node(c1, c2, I_e, u_e[t], u_i[t], u_e[t], d, 0, 0, 0))
        u_i.append(u_i[t] + dt*wc_sim_node(c3, c4, I_i, u_e[t], u_i[t], u_i[t], d, 0, 0, 0))
        time.append((t+1) * dt)
        
    plt.figure(1, figsize=(30, 6)) 
    plt.subplot(1, 3, 1)
    plt.title("Excitatory Neuron Population")
    plt.plot(time, u_e,'-k')
    plt.xlabel("Time (s)")
    
    plt.subplot(1, 3, 2)
    plt.plot(time,u_i)
    plt.title("Inhibitory Neuron Population")
    plt.xlabel("Time (s)")
    
#    plt.subplot(1, 3, 2)
#    plt.plot(time, u_e, '-k', linewidth=0.2)
#    plt.plot(time,u_i, '-k', linewidth=0.2)
#    plt.xlabel("Time (s)")

    plt.show()

def plot_cor_mat(data_array, nodes, skip, plot = False):
    corr_mat= []
    for i in range(nodes):
        temp = []
        for j in range(nodes):
            #only measure after time 20 - ignoring the intial stabilization of
            #time series
            temp.append(np.corrcoef(data_array[i][skip:],data_array[j][skip:])[0][1])
        corr_mat.append(temp)

    corr_mat = np.transpose(np.array(corr_mat))
    if plot:
        fig, ax = plt.subplots()
        plot2 = corr_mat
        plt.title("Correlation Matrix")
        np.fill_diagonal(plot2, 0)
        cax = ax.imshow(plot2, aspect='auto', vmax = 1, vmin = -1)
        cbar = fig.colorbar(cax)
        plt.show()
#         print(corr_mat)
    return corr_mat

def plot_pwr_spec(data_array, nodes, fs, dt = 0.01, length = all):
    for i in range(nodes):
#        f, Pxx_den = signal.welch(data_array[i], (1/dt*10))
        f, Pxx_den = signal.welch(data_array[i], fs)

        plt.semilogy(f, Pxx_den,'-k', linewidth=0.2)
        if length !=all:
            plt.xlim([0,length])
        plt.ylim(bottom = 10e-9)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Log Power')
        
    plt.show()
    
def get_corr_mat(g, speed):
    tau_mat = tau(T_new, speed, dt)
    ue_array, ui_array = wc_model_sim(g, tau_mat, W, True)
    cmat = plot_cor_mat(ue_array)
    return cmat

def plot_mat(matrix, title: str = "title", _max= None, _min: int = 0,
             exclude = None, half = False):
    """
    give a matrix of values plot a heat map
    """
    nodes = np.shape(matrix)[0]
    print(nodes)
    if _max == None:
        np.max(matrix)
    copy = matrix.copy()
    if exclude:
        copy[copy<= exclude] = 0
    if half:
        for row in range(nodes):
            for col in range(row+1, nodes):
                copy[row,col] = copy[col,row]
            
    fig, ax = plt.subplots()
    plt.title(title)
    cax = ax.imshow(copy, aspect='auto', vmax = _max, vmin = _min)
    cbar = fig.colorbar(cax)
    plt.show()
#    print(copy)
    
def plot_mat2(matrix, fig=None, ax= None,_max= None, _min: int = 0,
             exclude = None, half = False):
    """
    give a matrix of values plot a heat map
    """
    nodes = np.shape(matrix)[0]
    if _max == None:
        np.max(matrix)
    copy = matrix.copy()
    if exclude:
        copy[copy<= exclude] = 0
    if half:
        for row in range(nodes):
            for col in range(row+1, nodes):
                copy[row,col] = copy[col,row]
            
    cax = plt.imshow(copy, aspect='auto', vmax = _max, vmin = _min)
    plt.colorbar(cax)
    #plt.show()
#    print(copy)