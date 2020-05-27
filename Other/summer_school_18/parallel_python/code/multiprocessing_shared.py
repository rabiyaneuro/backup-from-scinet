# multiprocessing-shared.py
import time
from multiprocessing import Process
from multiprocessing import Value
def myfunc(v):
  for i in range(50):
    time.sleep(0.001)
    v.value += 1

v = Value('i', 0);
procs = []
for i in range(10):
  p=Process(target=myfunc,args=(v,))
  procs.append(p)
  p.start()
for proc in procs: proc.join()
print(v.value)
