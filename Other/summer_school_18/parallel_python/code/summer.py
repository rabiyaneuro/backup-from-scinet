# summer.py

try:
    xrange
except NameError:
    xrange = range
    
def my_summer(start, stop):
    total = 0
    for i in xrange(start, stop):
        total += i


def my_summer2(data):
    total = 0
    start, stop = data
    for i in xrange(start, stop):
        total += i
