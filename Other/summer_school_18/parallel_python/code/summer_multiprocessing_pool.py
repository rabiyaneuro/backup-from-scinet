
#file: summer_multiprocessing_pool.py
import time, multiprocessing
from summer import my_summer2

begin = time.time()

    # p = multiprocessing.Process(target = my_summer, args = (0, 5000000))
    # processes.append(p)
    # p.start()

numjobs = 10
numprocs = multiprocessing.cpu_count()

input = [(0, 5000000)] * numjobs
p = multiprocessing.Pool(processes = numprocs)
p.map(my_summer2, input)
#p.close()
#p.join()
#for p in processes: p.join()

print "Time:", time.time() - begin
