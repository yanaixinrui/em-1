#!/usr/bin/env python3

"""
ideas from 
https://docs.python.org/2/library/profile.html
https://amaral.northwestern.edu/resources/guides/speed-your-python-numpy-codes


"""
import numpy as np
import random

# performace counter, does not include time during sleep 
from time import process_time
import cProfile
import timeit

SIZE = 200


#initialize data
def initialize_data():
    random.seed(7)
    A = np.random.rand(SIZE,SIZE)
    B = np.random.rand(SIZE,SIZE)
    return A, B

#----------------------------------------------------------
# simple deep copy

def cp_for_for(source_mat, dest_mat):
    for i in range(SIZE):
        for j in range(SIZE):
            dest_mat[i,j] = source_mat[i,j]
    
def cp_row_for(source_mat, dest_mat):
    for i in range(SIZE):
            dest_mat[:,i] = source_mat[:,i]

def cp_col_for(source_mat, dest_mat):
    for i in range(SIZE):
            dest_mat[i,:] = source_mat[i,:]

def cp_mat(source_mat, dest_mat):
    dest_mat[:,:] = source_mat[:,:]


    
#----------------------------------------------------------
# inner product (matrix mult)

'''
            
def inner_row_for(source_mat_A, source_mat_B, dest_mat):
    for i in range(SIZE):
            dest_mat[:,i] = source_mat[:,i]

def inner_col_for(source_mat_A, source_mat_B, dest_mat):
    for i in range(SIZE):
            dest_mat[i,:] = source_mat[i,:]

'''

def inner_for_for(source_mat_A, source_mat_B, dest_mat):
    for i in range(SIZE):
        for j in range(SIZE):
            inner_sum = 0.
            for k in range(SIZE):
                inner_sum += source_mat_A[i,k] * source_mat_B[k,j]
            dest_mat[i,j] = inner_sum

def inner_mat(source_mat_A, source_mat_B, dest_mat):
    dest_mat[:,:] = np.dot(source_mat_A,source_mat_B)
    
#----------------------------------------------------------
# element mult (element by element multiply)

def plus_mult_setup():
    a = np.random.rand(SIZE,SIZE)
    b = np.random.rand(SIZE,SIZE)
    c = np.random.rand(SIZE,SIZE)
    return a,b,c

def plus_mult_inplace(a,b,c):
    a += b
    a *= c
    

def plus_mult_new(a,b,c):
    a = a*b + c
    
#start = process_time()

A,B = initialize_data()
a,b,c = plus_mult_setup()
C = np.zeros((SIZE,SIZE),dtype=float)

cProfile.run('inner_for_for(a,b,c)')
print(c)
#a,b,c = plus_mult_setup()
cProfile.run('inner_mat(a,b,c)')
print(c)
'''
cProfile.run('cp_for_for(A,C)')
C.fill(0.)
cProfile.run('cp_row_for(A,C)')
C.fill(0.)
cProfile.run('cp_col_for(A,C)')
C.fill(0.)
cProfile.run('cp_mat(A,C)')
'''
'''
cProfile.run('plus_mult_inplace(a,b,c)',globals(), locals())
cProfile.run('plus_mult_new(a,b,c)',globals(), locals())

cProfile.run('plus_mult_inplace(a,b,c)')
cProfile.run('plus_mult_new(a,b,c)')
'''


