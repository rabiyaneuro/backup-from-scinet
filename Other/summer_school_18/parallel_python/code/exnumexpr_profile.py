#!/usr/bin/env python
#
# exnumexpr-profile.py
#
# Ramses van Zon
# SciNetHPC, 2016
#
import numpy as np
import numexpr as ne

@profile
def main():
    a = np.random.rand(10*1000*1000)
    b = np.random.rand(10*1000*1000)
    c = np.zeros(10*1000*1000)
    c = a**2 + b**2 + 2*a*b
    old=ne.set_num_threads(1)
    ne.evaluate("a**2+b**2+2*a*b",out=c)
    old=ne.set_num_threads(2)
    ne.evaluate("a**2+b**2+2*a*b",out=c)

main()
    
