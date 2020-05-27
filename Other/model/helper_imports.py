# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:16:23 2018
All the imports needed to run the functions in helper_hpc.py

@author: Administrator
"""

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
from math import exp
import time


"""
 
seed_ = 2018

# only if running on scinet
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = int(comm.Get_rank())
#seed_ = int(2018+rank)

RNG = np.random.RandomState(seed_)
"""