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
from helper_hpc import *
import time
#%% Simulating a system with n nodes and setting constants

"""


g = 0.5
cond_vel = 4000
c1 = 1.6
c2 = -4.7
c3 = 3
c4 = -0.63
I_e = 1.8
I_i = -0.2
nodes = 6

# anatomical data
#W is the matrix from cocomac from tvb

#setting the seed
def weights(nodes):
    np.random.seed(2018) 
    
    W_mat = np.ones((nodes, nodes))
    np.fill_diagonal(W_mat, 0)
    
    # changing weights from 1 to random number 
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            W_mat[row,col] = W_mat[row, col] * np.random.randint(0,100)/100
            W_mat[col,row] = W_mat[row,col]
    return W_mat

W_mat = weights(nodes)
        

#b = np.random.randint(0,100,size=(6,6))/100
#b= (b + b.T)/2
#np.fill_diagonal(b, 0)

# tract lengths
#tract_mat = np.ones((nodes, nodes), dtype = int)

def tracts(nodes):
    np.random.seed(2018) 
    tract_mat = np.ones((nodes, nodes))
    np.fill_diagonal(tract_mat, 0)
    
    # changing weights from 1 to random number 
    for row in range(0,nodes):
        for col in range(row+1, nodes):
            tract_mat[row,col] = tract_mat[row, col] * np.random.randint(0,100)/100
            tract_mat[col,row] = tract_mat[row,col]
            
    return tract_mat

tract_mat = tracts(nodes)

"""
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
#            u_e_.append(u_e_tsteps[t][i] + dt*(wc_sim_node(c1, c2, I_e, u_e_tsteps[t][i], 
#                                                          u_i_tsteps[t][i], u_e_tsteps[t][i], d, g, W_mat[i,:], 
#                                                          (delays_ue))))   
#
#            u_i_.append(u_i_tsteps[t][i] + dt*(wc_sim_node(c3, c4, I_i, 
#                                                          u_e_tsteps[t][i], u_i_tsteps[t][i],u_i_tsteps[t][i], 
#                                                          d, 0, W_mat[i,:], (delays_ue))))
#
#        #append all node values to master list
#        u_e_tsteps.append(u_e_)
#        u_i_tsteps.append(u_i_)
#
#    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps))
#%% new
    

def wc_model_sim_new(g, cond_vel, c1, c2, c3, c4, I_e, I_i, nodes, tract_mat, W_mat,
                 dt = 0.01, time_steps = 2000, d= 0.001,  
                 seed_num = None, delay = True):
    if seed_num:
        rng_wc = np.random.RandomState(seed_num)
    else:
        rng_wc = np.random.RandomState(0)
    
    if not delay:
        delays_mat = np.zeros_like(tract_mat, dtype = int)
    else:
        delays_mat = tau(tract_mat, cond_vel, dt)

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
            
        u_e_new = u_e_tsteps[t] + dt*(wc_sim_node_new(c1, c2, I_e, u_e_tsteps[t], 
                            u_i_tsteps[t], u_e_tsteps[t], d, g, W_mat, (big_delay),
                            rng_wc))
        u_i_new = u_i_tsteps[t] + dt*(wc_sim_node_new(c3, c4, I_i, u_e_tsteps[t], 
                            u_i_tsteps[t],u_i_tsteps[t], d, 0, W_mat, (big_delay),
                            rng_wc))  


        #append all node values to master list
        u_e_tsteps.append(u_e_new)
        u_i_tsteps.append(u_i_new)

    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps))

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
#    
#%%
#np.random.seed(2018)
#ue_targ_new, ui_targ_new = wc_model_sim_new(g, cond_vel, tract_mat, W_mat,
#                                            c1, c2, c3, c4, I_e, I_i, nodes, d= 0.1)
#
#
#ue_targ_old, ui_targ_old = wc_model_sim(g, cond_vel, tract_mat, W_mat, c1, 
#                                        c2, c3, c4, I_e, I_i, nodes, d= 0.1)
#    
