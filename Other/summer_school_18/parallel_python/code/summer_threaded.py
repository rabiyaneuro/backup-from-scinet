# file: summer_threaded.py
import time, threading
from summer import my_summer

begin = time.time()
threads = []

for i in range(10):
    t = threading.Thread(target = my_summer, args = (0, 5000000))
    threads.append(t)
    t.start()

for t in threads: t.join()

print ("Time: %f"%(time.time() - begin))
