"""
copy of the helper file, this one has the optimized versions of the functions,
use this for scinet
"""
import sys

sys.path.insert(0, './Helper')

from helper_imports import *
#%%timing

def time2run(fxn, *args):
    t = time.time()
    fxn(*args)
    elap1 = time.time() - t
    print(elap1)
    
#%%
def p2matrix(p, nodes):
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
    print(copy)