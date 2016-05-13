#!/usr/bin/env python


"""
------------------------------------------------------------------------------
Classes to implement Gaussian mixture model with EM
------------------------------------------------------------------------------
"""
import numpy as np

#import itertools # for product
#import sys        # for sys.float_info.max
#from collections import defaultdict

import matplotlib.pyplot as plt
import unittest

try:
    from scipy.misc  import logsumexp
except ImportError:
    def logsumexp(x,**kwargs):
        return np.log(np.sum(np.exp(x),**kwargs))


#============================================================================
class Mm():
    """ 
    """

    def __init__(self, point_list=[[]]): 
        self.X = np.matrix(point_list,copy = True,dtype=float) # X[i,point]
        self.n,self.d = self.X.shape 
    
    def __str__(self):
        s = "mm data: (N = " + str(self.n) + ", D = " + str(self.d) +")"
        for point in self.X:
            s += "\n  " + str(point)
        return s
    
    def find_means(self, n_means):

        # initialize
        maxes = self.X.max(axis=0) # ndarray [d]
        mins = self.X.min(axis=0)  # ndarray [d]
        ranges = maxes-mins
        # Q: determine whether numpy compiles below statements into single cmd?
        means = np.random.rand(n_means,self.d) # [nm x d] mat
        means = np.multiply(ranges, means) # [d] *. [nm x d] 
        means = means + mins.T # [nm x d] + [1 x d]

        # calculate counts
        
                       
        return means
        
    
#============================================================================
class TestCrf(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(self):
        """ runs once before ALL tests """
        print("\n...........unit testing class Mm..................")
        #self.my_Crf = Crf()

    def setUp(self):
        """ runs once before EACH test """
        pass

    #@unittest.skip
    def test_init(self):
        print("\n...testing init(...)")
        a = [[1,2,3],[4,5,6]]
        m = Mm(a)
        print(m)
    
    #@unittest.skip
    def test_simple(self):
        with open("points.dat") as f:
            data_mat = []
            for line in f:
                sline = line.split()
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat)
        
        plt.scatter(mm.X[:,0],mm.X[:,1])
        means = mm.find_means(2)
        print(means)
        plt.scatter(means[:,0],means[:,1],s=300,c='r')
        plt.show()
        #plt.draw()
        print(mm)
        #pause = input("press enter when done")

    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class Mm complete..............\n")
        
#============================================================================
if __name__ == '__main__':
    unittest.main()
    
    
# 11:00
# 2:40 


       