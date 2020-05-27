# child.py
import os
print("Hello from", os.getpid())
os._exit(0)
