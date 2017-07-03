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

def numpy_math(value):
    a = numpy.sin(value)
    b = numpy.log(value)
    
    return numpy.dot(a,b)


# create a list of 8 large numpy arrays
a = [numpy.arange(1000000) for _ in range(16)]
pool = Pool(processes = 4)

start = time.time()
result = numpy.sin(a)
end = time.time()
print('Singled threaded %f' % (end - start))
start = time.time()

# Pool will call numpy_sin with a separate process with each list element as arg
result = pool.map(numpy_sin, a)
pool.close()
pool.join()
end = time.time()
print('Multithreaded %f' % (end - start))

start = time.time()
result = [numpy_math(a_i) for a_i in a]
end = time.time()
print('Singled threaded %f' % (end - start))
start = time.time()

# Pool will call numpy_sin with a separate process with each list element as arg
result = pool.map(numpy_math, a)
pool.close()
pool.join()
end = time.time()
print('Multithreaded - math %f' % (end - start))