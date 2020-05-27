#!/usr/bin/env python
# auc_serial.py
import sys
ntot  = int(sys.argv[1])
dx    = 3.0/ntot
width = 3.0
x = 0
a = 0.0
for i in range(ntot):
    y = 0.7*x**3 - 2*x**2 + 4
    a += y*dx
    x += dx
print("The area is %f"%a)

