#!/usr/bin/env python
# auc_numpy.py
import sys
import time
from mpi4py import MPI
from numpy import asarray, zeros
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
ntot = int(sys.argv[1])
ntot -= ntot%size
npnts = ntot//size
dx = 3.0/ntot
width = 3.0/size
x = rank*width
a = 0.0
for i in range(npnts):
    y = 0.7*x**3 - 2*x**2 + 4
    a += y*dx
    x += dx
    print(rank)
    time.sleep(5)
ans = zeros(1)
MPI.COMM_WORLD.Reduce(asarray(a),ans)
if rank == 0:
    print("The area is %f"%ans[0])

