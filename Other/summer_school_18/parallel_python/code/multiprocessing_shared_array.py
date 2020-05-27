# multiprocessing-shared-array.py
from numpy import arange
from multiprocessing import Process,Array
def myfunc(a, i):
  a[i] = -a[i]
arr = Array('d', arange(10.))
procs = []
for i in range(10):
  p = Process(target = myfunc, args = (arr, i))
  procs.append(p)
  p.start()
for proc in procs:
  proc.join()
print(arr[:])
