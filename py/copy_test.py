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

SIZE = 5000


#initialize data
def initialize_data():
    random.seed(7)
    A = np.random.rand(SIZE,SIZE)
    B = np.random.rand(SIZE,SIZE)
    return A, B

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


def dot_for_for(source_mat_A, source_mat_B, dest_mat):
    for i in range(SIZE):
        for j in range(SIZE):
            dest_mat[i,j] += source_mat_A[i,j] * source_mat_B[i,j]
    
def dot_row_for(source_mat, dest_mat):
    for i in range(SIZE):
            dest_mat[:,i] = source_mat[:,i]

def cp_col_for(source_mat, dest_mat):
    for i in range(SIZE):
            dest_mat[i,:] = source_mat[i,:]

def cp_mat(source_mat, dest_mat):
    dest_mat[:,:] = source_mat[:,:]
    
    
#start = process_time()

A,B = initialize_data()
C = np.zeros((SIZE,SIZE),dtype=float)
cProfile.run('cp_for_for(A,C)')
C.fill(0.)
cProfile.run('cp_row_for(A,C)')
C.fill(0.)
cProfile.run('cp_col_for(A,C)')
C.fill(0.)
cProfile.run('cp_mat(A,C)')


