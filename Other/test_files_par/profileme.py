# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:46:22 2018

This is test script that I'm using to profile the function wc_model_sim() and
see where most of the time is spent. I will be using line_profiler to do this

@author: Administrator
"""

import numpy as np

g = 0.4
cond_vel = 4000
c1 = 1.6
c2 = -4.7
c3 = 3
c4 = -0.63
I_e = 1.8
I_i = -0.2
nodes = 6
speed = 4

# anatomical data
#W is the matrix from cocomac from tvb
np.random.seed(2018)
W_mat = np.ones((nodes, nodes))
np.fill_diagonal(W_mat, 0)

# changing weights from 1 to random number 
for row in range(0,nodes):
    for col in range(row+1, nodes):
        W_mat[row,col] = W_mat[row, col] * np.random.randint(0,100)/100
        W_mat[col,row] = W_mat[row,col]
        
"""
b = np.random.randint(0,100,size=(6,6))/100
b= (b + b.T)/2
np.fill_diagonal(b, 0)

"""

# tract lengths
tract_mat = np.ones((nodes, nodes))
#%% Target data
def mult(ar1, ar2):
    assert np.shape(ar1) == np.shape(ar2), "not same dim"
    res = 0
    for i in range(np.shape(ar1)[0]):
        res = res + ar1[i]*ar2[i]
    return res

def sig(u):
     return (1/(1 + np.exp(-50*u)))

def tau(t,c, timestep):
    '''
    t: tract length matrix
    c: conduction speed
    timestep: timestep used to solve ode
    
    returns an int representing the lag in # of timesteps
    '''
    return (np.round((t/c)/timestep)).astype(int)

def wc_sim_node(ce, ci, I, u_e, u_i, u, d, connec_strength, weights_mat, delays_mat):
    '''
    u is either u_e or u_i
    ce and ci are c1,c2,c3 or c4
    I is either I_e or I_i
    '''
    base = -u + ce*sig(u_e) + ci*sig(u_i) + I
    
    noise = np.sqrt(2*d)*np.random.normal(0, 1, size=1)[0]
#     noise =0
    if (connec_strength!= 0):
#         print(delays_mat, np.shape(delays_mat), sig(delays_mat), np.shape(sig(delays_mat)))
        connec = connec_strength*(mult(weights_mat, sig(np.array(delays_mat))))
        return base + noise + connec
    
    return base + noise

np.random.seed(2018)

@profile
def wc_model_sim(g, cond_vel, tract_mat, W_mat, c1, c2, c3, c4, I_e, I_i, nodes,
                 dt = 0.01, time_steps = 2000, d= 0.001, delay = False):
    isRand = True
    
    if not delay:
        delays_mat = np.zeros_like(tract_mat)
    else:
        delays_mat = tau(tract_mat, cond_vel, dt)

    u_e_tsteps = [] #will be a list of size 'max_delay' vectors each of size 'nodes'
    u_i_tsteps = []

    #considering the case when no delays
    if (np.max(delays_mat) ==0):
        max_delay = 1
    else:
        max_delay = np.max(delays_mat)
    
    if (isRand == False):
    #fill arrays with constant until max delay
        for i in range(max_delay):
            ue_0 = []
            ui_0 = []
            for i in range(nodes):
                ue_0.append(0)
                ui_0.append(0)
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)
       
    else:
    #fill arrays with randoms until max delay
        for i in range(max_delay):
            ue_0 = []
            ui_0 = []
            for i in range(nodes):
                ue_0.append(np.random.randint(-25,25)/100)
                ui_0.append(np.random.randint(-25,25)/100)
            np.random.shuffle(ui_0)
            np.random.shuffle(ue_0)
            u_e_tsteps.append(ue_0)
            u_i_tsteps.append(ui_0)
    
    
    for t in range(max_delay-1, time_steps):
        u_e_ = []
        u_i_ = []
        

        for i in range(nodes):
            
            #making the delays vector that contains u values at delayed t
            delays_ue = []
            delays_ui = []
            for other_node in range(nodes):
                if not delay:
                    del_t = 0
                else:
                    del_t = delays_mat[i,other_node]
                delays_ue.append(u_e_tsteps[t-del_t][other_node]) 
                delays_ui.append(u_i_tsteps[t-del_t][other_node])

            #calculate the ue and ui value for node i
            
            
            u_e_.append(u_e_tsteps[t][i] + dt*(wc_sim_node(c1, c2, I_e, u_e_tsteps[t][i], 
                                                          u_i_tsteps[t][i], u_e_tsteps[t][i], d, g, W_mat[i,:], 
                                                          (delays_ue))))       
            u_i_.append(u_i_tsteps[t][i] + dt*(wc_sim_node(c3, c4, I_i, 
                                                          u_e_tsteps[t][i], u_i_tsteps[t][i],u_i_tsteps[t][i], 
                                                          d, 0, W_mat[i,:], (delays_ui))))
            
#             # no delays
#             u_e_.append(u_e_tsteps[t][i] + dt*(wc_sim_node(c1, c2, I_e, u_e_tsteps[t][i], 
#                                                           u_i_tsteps[t][i], u_e_tsteps[t][i], d, g, W_mat[i,:], 
#                                                           u_e_tsteps[t])))       
#             u_i_.append(u_i_tsteps[t][i] + dt*(wc_sim_node(c3, c4, I_i, 
#                                                           u_e_tsteps[t][i], u_i_tsteps[t][i],u_i_tsteps[t][i], 
#                                                           d, 0, W_mat[i,:], u_e_tsteps[t])))


        #append all node values to master list
        u_e_tsteps.append(u_e_)
        u_i_tsteps.append(u_i_)
                       
    return np.transpose(np.array(u_e_tsteps)), np.transpose(np.array(u_i_tsteps)) 

wc_model_sim(g, cond_vel, tract_mat, W_mat, c1, c2, c3, c4, I_e, I_i, nodes, d= 0.1)