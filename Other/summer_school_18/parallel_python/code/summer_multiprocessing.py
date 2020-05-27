
#file: summer_multiprocessing.py
import time, multiprocessing
from summer import my_summer

begin = time.time()
processes = []

for i in range(10):
    p = multiprocessing.Process(target = my_summer, args = (0, 5000000))
    processes.append(p)
    p.start()

for p in processes: p.join()

print ("Time:%f"%(time.time() - begin))
