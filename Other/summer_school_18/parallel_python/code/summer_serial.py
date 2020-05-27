# file: summer_serial.py
import time
from summer import my_summer

begin = time.time()
threads = []

for i in range(10):
    my_summer(0, 5000000)

print ("Time: %f"%(time.time() - begin))
