# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:24:52 2018

This file contains two versions of the function wc_node_sim. One is more
 optimized than the other, they both return the same results save for a floating
 point precision accuracy discrepency. If you round the results of both the 
 functions then the results are equal.
 
TO DO: work on precision of of floats


@author: Administrator
"""
import helper_functions as hf
import numpy as np
from memory_profiler import profile

#%% Kuramoto model

@profile
def km_model_sim_d(km_params: dict, delays_mat, max_delay, W_mat, nodes, seed_num = 123):
    """
    km_params : dictionary containing wi, K, N, timesteps, dt, constant

    """
    w_i, dt, time_steps = km_params['wi'], km_params['dt'], km_params['time_steps']
    
    N, K = km_params['N'], km_params['K']
    
    constant = km_params['constant']
    

    theta_tsteps = [] #will be a list of  vectors (lenght = 'max_delay') 
                    # each of size 'nodes'


    if not constant:
        #print("not constant")
        rng_wc = np.random.RandomState(seed_num)

        for i in range(max_delay+2):
            ue_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            
            theta_tsteps.append(ue_0)
            
    else:
        #print("constant")
        for i in range(max_delay+2):
            ue_0 = np.arange(nodes)*0.0
    
            theta_tsteps.append(ue_0)

    for t in range(max_delay+1, max_delay+1+time_steps):
        # this is the list that theta value of each node i, at time t
        theta_i_t = []
        
        for i in range(nodes):
            input_i = 0 # the value of the input into theta at node i
            
            # adding up the input from each other node j
            for j in range(nodes):
                tau_ij = delays_mat[i,j]
#                print(tau_ij)
                
                # the value at row j and column (t-tau_ij)
#                print(theta_tsteps)
                theta_j_delay = theta_tsteps[t-tau_ij][j]
               
                input_i = input_i + np.sin(theta_j_delay- theta_tsteps[t][i])
            
            # the value of theta at node i and time t+1
            theta_i = (w_i[i] + (K/N)* input_i)*dt + theta_tsteps[t][i]
            theta_i_mod = np.mod(theta_i, 2*np.pi)
            # append it to the full list of theta's at time t+ 1
            theta_i_t.append(theta_i_mod)
        
        #print(np.shape(theta_i_t), "shape of theta_i_t")
        theta_tsteps.append(theta_i_t)
        #print(np.shape(theta_tsteps))
    return np.transpose(np.array(theta_tsteps))


#%% new
def wc_modelsim_c(wc_params: dict, tract_mat, c_mat, nodes, seed_num):
    """
    wc_params : dictionary containing g, c1, c2, c3, c4, I_e, I_i, 
    dt, d, timesteps
    
    c_mat: is a element-by-element reciprocal matrix of the conduction velocities
    
    """
    dt, time_steps = wc_params['dt'], wc_params['time_steps']
    
    c1, c2, c3, c4, d, I_e, I_i, g = wc_params['c1'], wc_params['c2'], \
    wc_params['c3'], wc_params['c4'], wc_params['d'], wc_params['I_e'], \
    wc_params['I_i'], wc_params['g']
    
    constant = wc_params['constant']
    
    #construct delays matrix - made up of int time steps
    #c_mat is 1/c matrix
    
    delays_mat = np.round(np.multiply((tract_mat/dt),c_mat)).astype(int)
    max_delay = np.max(delays_mat)
    
    rng_wc = np.random.RandomState(seed_num)
    
    u_e_tsteps = [] #will be a list of size 'max_delay' vectors each of size 'nodes'
    u_i_tsteps = []

#    print(max_delay, "MAXD")
#    print(delays_mat, "DMAT")
    #fill arrays with constant until max delay, the +2 is if there's 0 delay, 
    # to fill it up with something
    print(max_delay)
    if not constant:
        print("not constant")
        for i in range(max_delay+2):
            ue_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            ui_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            
    
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)
            
    else:
        print("constant")
        for i in range(max_delay+2):
            ue_0 = np.arange(nodes)*0.0
            ui_0 = np.arange(nodes)*0.0
    
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)
        
    # u_e_tsteps is a list of arrays
        
    #starting at max_delay +1 because there is currently max_delay+2 elements 
    #in array and to start at the last element you need to start at max_delay+1
    
    # for each time step (column in u_e_tsteps)
    for t in range(max_delay+1, time_steps):
        big_delay = [] # will hold a matrix of values, for 

        for i in range(nodes):            
            delays_ue = []
#            delays_ui = []

            for other_node in range(nodes):
                del_t = delays_mat[i,other_node]
#                print(len(u_e_tsteps), t-del_t, other_node)
                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
#                delays_ui.append(u_i_tsteps[t-del_t][other_node])
            big_delay.append(delays_ue)
#            print(delays_ue)
#        print(big_delay, "BIGDE")
        
        u_e_new = u_e_tsteps[t] + dt*(hf.wc_sim_node_no_w(c1, c2, I_e, u_e_tsteps[t], 
                            u_i_tsteps[t], u_e_tsteps[t], d, g, (big_delay),
                            rng_wc))
        u_i_new = u_i_tsteps[t] + dt*(hf.wc_sim_node_no_w(c3, c4, I_i, u_e_tsteps[t], 
                            u_i_tsteps[t],u_i_tsteps[t], d, 0, (big_delay),
                            rng_wc))  


        #append all node values to master list
        u_e_tsteps.append(u_e_new)
        u_i_tsteps.append(u_i_new)

    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps)), delays_mat

#%%
def wc_model_sim_new(wc_params: dict, tract_mat, c_mat, W_mat, nodes, seed_num,
                     lowest_c, delay = True):
    """
    wc_params : dictionary containing g, cond_vel, c1, c2, c3, c4, I_e, I_i, 
    dt, d, timesteps
	
	c_mat is actually 1/c where c is the conduction velocity in mm/s
    """
    dt, time_steps = wc_params['dt'], wc_params['time_steps']
    
    c1, c2, c3, c4, d, I_e, I_i, g = wc_params['c1'], wc_params['c2'], \
    wc_params['c3'], wc_params['c4'], wc_params['d'], wc_params['I_e'], \
    wc_params['I_i'], wc_params['g']
    
    constant = wc_params['constant']
    
    rng_wc = np.random.RandomState(seed_num)

    if not delay:
        delays_mat = np.zeros_like(tract_mat, dtype = int)
    else:
        delays_mat = np.round(np.multiply((tract_mat/dt),c_mat)).astype(int)
    #print(delays_mat)

    u_e_tsteps = [] #will be a list of size 'max_delay' vectors each of size 'nodes'
    u_i_tsteps = []
    
    #print(delays_mat)

    #considering the case when delays
    max_delay = np.round(np.multiply((np.max(tract_mat)/dt),lowest_c)).astype(int)
    #print("max", max_delay)
	
	
    # #considering the case when no delays
    # if (np.max(delays_mat) ==0):
        # max_delay = 1
  
    #setting the seed
#    np.random.seed(2018)
    #print(rng_wc.get_state()[1][0])

    if not constant:
        #print("not constant")
        for i in range(max_delay+2):
            ue_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            ui_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            #print(ue_0,ui_0)
    
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)
            
    else:
        #print("constant")
        for i in range(max_delay+2):
            ue_0 = np.arange(nodes)*0.0
            ui_0 = np.arange(nodes)*0.0
    
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)

    for t in range(max_delay+1,max_delay+1 + time_steps):
        big_delay = []

        for i in range(nodes):            
            delays_ue = []
            delays_ui = []

            for other_node in range(nodes):
                if not delay:
                    del_t = 0
                else:
                    del_t = delays_mat[i,other_node]
                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
                delays_ui.append(u_i_tsteps[t-del_t][other_node])
            big_delay.append(delays_ue)
            
        u_e_new = u_e_tsteps[t] + dt*(hf.wc_sim_node_new(c1, c2, I_e, u_e_tsteps[t], 
                            u_i_tsteps[t], u_e_tsteps[t], d, (1/nodes)*g, W_mat, (big_delay),
                            rng_wc))
        u_i_new = u_i_tsteps[t] + dt*(hf.wc_sim_node_new(c3, c4, I_i, u_e_tsteps[t], 
                            u_i_tsteps[t],u_i_tsteps[t], d, 0, W_mat, (big_delay),
                            rng_wc))  


        #append all node values to master list
        u_e_tsteps.append(u_e_new)
        u_i_tsteps.append(u_i_new)

    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps)), max_delay

#%%
def wc_model_sim_d(wc_params: dict, delays_mat, max_delay, W_mat, nodes, seed_num):
    """
    wc_params : dictionary containing g, cond_vel, c1, c2, c3, c4, I_e, I_i, 
    dt, d, timesteps

    """
    dt, time_steps = wc_params['dt'], wc_params['time_steps']
    
    c1, c2, c3, c4, d, I_e, I_i, g = wc_params['c1'], wc_params['c2'], \
    wc_params['c3'], wc_params['c4'], wc_params['d'], wc_params['I_e'], \
    wc_params['I_i'], wc_params['g']
    
    constant = wc_params['constant']
    
    rng_wc = np.random.RandomState(seed_num)

    u_e_tsteps = [] #will be a list of size 'max_delay' vectors each of size 'nodes'
    u_i_tsteps = []


    if not constant:
        #print("not constant")
        for i in range(max_delay+2):
            ue_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            ui_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
            
    
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)
            
    else:
        print("constant")
        for i in range(max_delay+2):
            ue_0 = np.arange(nodes)*0.0
            ui_0 = np.arange(nodes)*0.0
    
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)

    for t in range(max_delay+1, max_delay+1+time_steps):
        big_delay = []

        for i in range(nodes):            
            delays_ue = []
            delays_ui = []

            for other_node in range(nodes):
                del_t = delays_mat[i,other_node]
                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
                delays_ui.append(u_i_tsteps[t-del_t][other_node])
            big_delay.append(delays_ue)
            
        u_e_new = u_e_tsteps[t] + dt*(hf.wc_sim_node_new(c1, c2, I_e, u_e_tsteps[t], 
                            u_i_tsteps[t], u_e_tsteps[t], d, g, W_mat, (big_delay),
                            rng_wc))
        u_i_new = u_i_tsteps[t] + dt*(hf.wc_sim_node_new(c3, c4, I_i, u_e_tsteps[t], 
                            u_i_tsteps[t],u_i_tsteps[t], d, 0, W_mat, (big_delay),
                            rng_wc))  


        #append all node values to master list
        u_e_tsteps.append(u_e_new)
        u_i_tsteps.append(u_i_new)

    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps)), delays_mat

#%% (w/ out dictionary as input)
def wc_model_sim_new_old(g, cond_vel, c1, c2, c3, c4, I_e, I_i, nodes, tract_mat, W_mat,
                 dt = 0.01, time_steps = 2000, d= 0.001,  
                 seed_num = None, delay = True):
    if seed_num:
        rng_wc = np.random.RandomState(seed_num)
    else:
        rng_wc = np.random.RandomState(0)
    
    if not delay:
        delays_mat = np.zeros_like(tract_mat, dtype = int)
    else:
        delays_mat = hf.tau(tract_mat, cond_vel, dt)

    u_e_tsteps = [] #will be a list of size 'max_delay' vectors each of size 'nodes'
    u_i_tsteps = []

    #considering the case when no delays

    #considering the case when no delays
    if (np.max(delays_mat) ==0):
        max_delay = 1
    else:
        max_delay = np.max(delays_mat)
  
    #setting the seed
#    np.random.seed(2018)
    print(rng_wc.get_state()[1][0])

    
    #fill arrays with constant until max delay
    for i in range(max_delay):
        ue_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
        ui_0 = rng_wc.randint(low = -25, high = 25, size = nodes)/100
        

        u_e_tsteps.append(ue_0)
        u_i_tsteps.append(ui_0)

    for t in range(max_delay-1, time_steps):
        big_delay = []

        for i in range(nodes):            
            delays_ue = []
            delays_ui = []

            for other_node in range(nodes):
                if not delay:
                    del_t = 0
                else:
                    del_t = delays_mat[i,other_node]
                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
                delays_ui.append(u_i_tsteps[t-del_t][other_node])
            big_delay.append(delays_ue)
            
        u_e_new = u_e_tsteps[t] + dt*(hf.wc_sim_node_new(c1, c2, I_e, u_e_tsteps[t], 
                            u_i_tsteps[t], u_e_tsteps[t], d, g, W_mat, (big_delay),
                            rng_wc))
        u_i_new = u_i_tsteps[t] + dt*(hf.wc_sim_node_new(c3, c4, I_i, u_e_tsteps[t], 
                            u_i_tsteps[t],u_i_tsteps[t], d, 0, W_mat, (big_delay),
                            rng_wc))  


        #append all node values to master list
        u_e_tsteps.append(u_e_new)
        u_i_tsteps.append(u_i_new)

    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps))    
#%%old
#
#def wc_model_sim(g, cond_vel, tract_mat, W_mat, c1, c2, c3, c4, I_e, I_i, nodes,
#                 dt = 0.01, time_steps = 2000, d= 0.001, delay = True):
#
#    
#    if not delay:
#        delays_mat = np.zeros_like(tract_mat, dtype = int)
#    else:
#        delays_mat = tau(tract_mat, cond_vel, dt)
#
#    u_e_tsteps = [] #will be a list of lists to hold the value of each node per time step
#    u_i_tsteps = []
#
#    #considering the case when no delays
#
#    #considering the case when no delays
#    if (np.max(delays_mat) ==0):
#        max_delay = 1
#    else:
#        max_delay = np.max(delays_mat)
#  
#    #setting the seed
##    np.random.seed(2018) 
#
#    #fill arrays with constant until max delay
#    for i in range(max_delay):
#        ue_0 = np.random.randint(low = -25, high = 25, size = nodes)/100
#        ui_0 = np.random.randint(low = -25, high = 25, size = nodes)/100
#        
#        u_e_tsteps.append(ue_0)
#        u_i_tsteps.append(ui_0)
#
#
#    for t in range(max_delay-1, time_steps):
#        u_e_ = [] # temp array to hold values at each node for the current time step t
#        u_i_ = []
#        
#        for i in range(nodes):
#            
#            delays_ue = []
#            delays_ui = []
#
#            for other_node in range(nodes):
#                if not delay:
#                    del_t = 0
#                else:
#                    del_t = delays_mat[i,other_node]
#                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
#                delays_ui.append(u_i_tsteps[t-del_t][other_node])
#
#            u_e_.append(u_e_tsteps[t][i] + dt*(hf.wc_sim_node(c1, c2, I_e, u_e_tsteps[t][i], 
#                                                          u_i_tsteps[t][i], u_e_tsteps[t][i], d, g, W_mat[i,:], 
#                                                          (delays_ue))))   
#
#            u_i_.append(u_i_tsteps[t][i] + dt*(hf.wc_sim_node(c3, c4, I_i, 
#                                                          u_e_tsteps[t][i], u_i_tsteps[t][i],u_i_tsteps[t][i], 
#                                                          d, 0, W_mat[i,:], (delays_ue))))
#
#        #append all node values to master list
#        u_e_tsteps.append(u_e_)
#        u_i_tsteps.append(u_i_)
#
#    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps))


#%%using numexpr
#def wc_model_sim_numexpr(g, cond_vel, tract_mat, W_mat, c1, c2, c3, c4, I_e, I_i, nodes,
#                 dt = 0.01, time_steps = 2000, d= 0.001, delay = True):
#
#    if not delay:
#        delays_mat = np.zeros_like(tract_mat, dtype = int)
#    else:
#        delays_mat = tau(tract_mat, cond_vel, dt)
#
#    u_e_tsteps = [] #will be a list of size 'max_delay' vectors each of size 'nodes'
#    u_i_tsteps = []
#
#    #considering the case when no delays
#
#    max_delay = 1
#  
#    #setting the seed
##    np.random.seed(2018)
#    
#    #fill arrays with constant until max delay
#    for i in range(max_delay):
#        ue_0 = np.random.randint(low = -25, high = 25, size = nodes)/100
#        ui_0 = np.random.randint(low = -25, high = 25, size = nodes)/100
#        
#
#        u_e_tsteps.append(ue_0)
#        u_i_tsteps.append(ui_0)
#
#    for t in range(max_delay-1, time_steps):
#        big_delay = []
#
#        for i in range(nodes):            
#            delays_ue = []
#            delays_ui = []
#
#            for other_node in range(nodes):
#                if not delay:
#                    del_t = 0
#                else:
#                    del_t = delays_mat[i,other_node]
#                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
#                delays_ui.append(u_i_tsteps[t-del_t][other_node])
#            big_delay.append(delays_ue)
#        
#        stuff1 = dt*(wc_sim_node_numexpr(c1, c2, I_e, u_e_tsteps[t], 
#                            u_i_tsteps[t], u_e_tsteps[t], d, g, W_mat, (big_delay) )) 
#        stuff1b = u_e_tsteps[t]
#        u_e_new = ne.evaluate("stuff1b + stuff1")
#        
#        stuff2 = dt*(wc_sim_node_numexpr(c3, c4, I_i, u_e_tsteps[t], 
#                            u_i_tsteps[t],u_i_tsteps[t], d, 0, W_mat, (big_delay)))
#        
#        stuff2b = u_i_tsteps[t]
#        u_i_new = ne.evaluate("stuff2b +   stuff2")
#
#
#        #append all node values to master list
#        u_e_tsteps.append(u_e_new)
#        u_i_tsteps.append(u_i_new)
#
#    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps))
    

#%%
# import time
#def test_same():
##    np.random.seed(2018)
#    ue_targ_new, ui_targ_new = wc_model_sim_new(g, cond_vel, tract_mat, W_mat,
#                                                c1, c2, c3, c4, I_e, I_i, nodes, d= 0.1)
#    
#    
#    ue_targ_old, ui_targ_old = wc_model_sim(g, cond_vel, tract_mat, W_mat, c1, 
#                                            c2, c3, c4, I_e, I_i, nodes, d= 0.1)
#    
#    return np.round(ue_targ_new,3) == np.round(ue_targ_old,3)
#
##%%
#
#def time2run(fxn, *args):
#    t = time.time()
#    fxn(*args)
#    elap1 = time.time() - t
#    print(elap1)
