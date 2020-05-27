#file: create12GB.py
#this creates a 12G/100 file
import numpy as np
n = 40000
fp = np.memmap('bigfile', dtype='float64', mode='w+', shape=(n,n))
print ("initial fragment:"); print (fp[0,0:5])
fp[0,:] = np.random.rand(n)
print ("random fragment:"); print (fp[0,0:5])
for i in xrange(n): fp[i,:] = np.random.rand(n)
