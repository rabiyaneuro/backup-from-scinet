# multiprocessing.shared-fixed.py
import time
from multiprocessing import Process, Value, Lock
def myfunc(v, lock):
  for i in range(50):
    time.sleep(0.001)
    with lock:
	print(v.value)
        v.value += 1
v = Value('i', 0)
lock = Lock()
procs = []
for i in range(10):
  p=Process(target=myfunc,args=(v, lock))
  procs.append(p)
  p.start()
for proc in procs: proc.join()
print(v.value)
