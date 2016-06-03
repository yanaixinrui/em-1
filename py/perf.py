#!/usr/bin/env python3

"""
original code copied  from 
http://stackoverflow.com/questions/11442191/parallelizing-a-numpy-vector-operation
"""
import time
import numpy
from multiprocessing import Pool

def numpy_sin(value):
    return numpy.sin(value)

a = [numpy.arange(10000000) for _ in range(8)]
pool = Pool(processes = 4)

start = time.time()
result = numpy.sin(a)
end = time.time()
print('Singled threaded %f' % (end - start))
start = time.time()
result = pool.map(numpy_sin, a)
pool.close()
pool.join()
end = time.time()
print('Multithreaded %f' % (end - start))