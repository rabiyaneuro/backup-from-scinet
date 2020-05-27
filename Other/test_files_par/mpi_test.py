#!/usr/bin/python
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import sys
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print (rank, size)
print (np.random.rand())


class A(object):

    def __init__(self, b=None, pops=6):
        self.pops = np.ones((6, 6)) * 1.0
        self.nump = pops
        self.energ = np.zeros((6, 6))
        self.iterb = b

    def nex(self):
        self.pops[-1] = self.pops[-1] * rank
        data_send = self.pops[-1]
        if rank == 0:
            comm.Send(data_send, dest=1, tag=13)
            data2 = np.empty(6, dtype=np.float64)
            comm.Recv(data2, source=size - 1, tag=13)
            self.pops[0] = data2
        elif rank == size - 1:
            comm.Send(data_send, dest=0, tag=13)
            data2 = np.empty(6, dtype=np.float64)
            comm.Recv(data2, source=size - 2, tag=13)
            self.pops[0] = data2
        else:
            comm.Send(data_send, dest=rank + 1, tag=13)
            data2 = np.empty(6, dtype=np.float64)
            comm.Recv(data2, source=rank - 1, tag=13)
            self.pops[0] = data2
        print(self.pops)


thing = A(1)
thing.nex()
"""





    def solve(self):
        for i in range(self.iterb):
            self.nex()
	print (self.pops)

    def nex_old(self):

# for each row in energ (for each pop), update according to pops

        nump_proc = int(self.nump / size)
        for i in range(rank * nump_proc, rank * nump_proc + nump_proc):
            self.energ[i] = self.energ[i] + 2 #updating by row
	    if rank != 0:
    		data = self.energ[i]
    		comm.Send(data, dest=1, tag=13)
	    if rank ==0:
		for rnk in range(1,size):
	            data = np.empty(6, dtype=np.float64)
    		    comm.Recv(data, source=rnk, tag=13)
		    self.energ[i] = data

spawn diff procs
each:
intialize class object
solve-
for each generation
loop through the appropriate energ in the array and update them

rank 0 = collect all energs- send it to the other arrays 

start new gen

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
ans = zeros(1)
MPI.COMM_WORLD.Reduce(asarray(a),ans)
if rank == 0:
    print("The area is %f"%ans[0])
"""
