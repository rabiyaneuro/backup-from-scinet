#!/usr/bin/env python
# auc.py
import sys
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
if len(sys.argv) == 2:
  ntot = int(sys.argv[1])
else:
  ntot = 10
if rank<size-1:
  npnts = ntot//size
else:
  npnts = ntot - (size-1)*(ntot//size)
dx = 3.0/ntot
width = 3.0/size
x = rank*width
a = 0.0
for i in range(npnts):
    y = 0.7*x**3 - 2*x**2 + 4
    a += y*dx
    x += dx
answer = MPI.COMM_WORLD.reduce(a)
if rank == 0:
    print("The area is %f"%answer)

