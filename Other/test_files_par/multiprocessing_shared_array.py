# multiprocessing-shared-array.py
from multiprocessing import Process,Array
import numpy as np

class A(object):
	def __init__(self,iter=None):
		self.pops = np.ones((4,4))
		self.iter = iter		
	
	def next(self, energ, i):
		peinr("here")
		print(i)
		energ[i] = sum(self.pops[i])
	
	def solve(self):
		arr = Array('d', np.arange(4.))
		procs = []

		for i in range(self.iter):
  			p = Process(target = next, args = (arr, i))
  			procs.append(p)
  			p.start()


			for proc in procs:
  				proc.join()
			print(arr[:])


	
	


thing = A(5)
thing.solve()


