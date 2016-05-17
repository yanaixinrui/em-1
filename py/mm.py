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
import time

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
        self.X = np.matrix(point_list,copy = True, dtype=float) # X[i,point]
        self.n,self.d = self.X.shape 
        #self.n_means = 1
        #self.means = np.mat(np.zeros((n_means, self.d),float))
    
    def __str__(self):
        s = "mm data: (N = " + str(self.n) + ", D = " + str(self.d) +")"
        for point in self.X:
            s += "\n  " + str(point)
        return s
    
    def k_means(self, n_means, n_iter=2):
        """ Lloyd's algorithm for solving k-means """
        
        # initialize
        maxes = self.X.max(axis=0) # ndarray [d]
        mins = self.X.min(axis=0)  # ndarray [d]
        ranges = maxes-mins
        
        # Q: determine whether numpy compiles below statements into single cmd?
        means = np.random.rand(n_means,self.d) # [nm x d] mat
        means = np.multiply(ranges, means) # [d] *. [nm x d] 
        means = means + mins # [nm x d] + [1 x d]
        
        self.plot_means(means)

        for i in range(n_iter):
            # given the current means, calculate counts for each
            count_of_each_mean = np.zeros((n_means,1), dtype=float)
            point_sum = np.mat(np.zeros((n_means,self.d), dtype=float))
            
            for point in self.X: # point is a row [1 x 2]
                # the "probability mass" is just inverse distance sq
                difference =  (means - point)          # [nm x d] - [1 x d]
                dist_sq = np.sum(np.multiply(difference, difference), axis=1)
                prob_mass = np.reciprocal(dist_sq)  # [2 x 1]
                max_mean_i = np.argmax(prob_mass)
                count_of_each_mean[max_mean_i] += 1 
                point_sum[max_mean_i] += point
                    
            #          [n_means x d] / [n_means x 1]
            new_means = point_sum / count_of_each_mean 
            means = new_means
            self.plot_means(means)

        return new_means

    def em(self, k, n_iter=2):
        """ Gmm 
            k is the number of Gaussian mixtures, meaning we have k
            separate Gaussian distributions, each represented by its own
            mean and variance.
            The graphical models is to roll a k-sided die --> 
              then select a sample from the k'th distribution """
            
        
        # initialize means with k_means
        means = self.k_means(k,5)
        sigmas = []
        pi = np.mat(np.ones(k,1))*(1/k) # start w uniform distribution
        self.plot_means(means)

        for i in range(n_iter):
            # given the current parameters, calc MLE prob mass for each mean
            prob_mass_of_each_k = np.zeros((k,1), dtype=float)
            weighted_sum_of_each_mean = np.mat(np.zeros((k,self.d), dtype=float))
            
            for point in self.X: # point is a row [1 x d]
                # 
                for k_i in range(k):
                    prob_mass =  pi[k_i] * 
                    prob_mass_of_each_k[k_i] += prob_mass 
                    point_sum[max_mean_i] += point
        
            # given counts, update means (normalize)
            #total_count = np.sum(count_of_each_mean)
            #fraction_of_each_mean = count_of_each_mean/total_count
            #new_means = fraction_of_each_mean * weighted_sum
            
            #          [n_means x d] / [n_means x 1]
            new_means = point_sum / prob_mass_of_each_k 
            means = new_means
            #print(means)
            self.plot_means(means)

        return new_means
        
    def plot_means(self, means):
        plt.clf()
        plt.scatter(np.asarray(self.X[:,0]), np.asarray(self.X[:,1]))
        plt.scatter(np.asarray(means[:,0]), np.asarray(means[:,1]), s=300,c='r')
        #plt.show()
        #plt.draw()
        plt.pause(0.01)
    

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

    @unittest.skip
    def test_init(self):
        print("\n...testing init(...)")
        a = [[1,2,3],[4,5,6]]
        m = Mm(a)
        print(m)
    
    @unittest.skip
    def test_k_means_simple(self):
        print("\n...test_k_means_simple(...)")
        data_mat = np.mat('1 1; 2 2; 1.1 1.1; 2.1,2.1')
        mm = Mm(data_mat)
        #print(mm)
        
        means = mm.k_means(2)
        #print(means)
        #mm.plot_means(means)
        
        self.assertTrue(((means[0,0] == 1.05) and (means[0,1] == 1.05)) or \
                   ((means[1,0] == 1.05) and (means[1,1] == 1.05)) )
        self.assertTrue(((means[0,0] == 2.05) and (means[0,1] == 2.05)) or \
                   ((means[1,0] == 2.05) and (means[1,1] == 2.05)) )

    def test_em_simple(self):
        print("\n...test_simple(...)")
        data_mat = np.mat('1 1; 2 2; 1.1 1.1; 2.1,2.1')
        mm = Mm(data_mat)
        #print(mm)
        
        means = mm.em(2)
        #print(means)
        #mm.plot_means(means)
        
        self.assertTrue(((means[0,0] == 1.05) and (means[0,1] == 1.05)) or \
                   ((means[1,0] == 1.05) and (means[1,1] == 1.05)) )
        self.assertTrue(((means[0,0] == 2.05) and (means[0,1] == 2.05)) or \
                   ((means[1,0] == 2.05) and (means[1,1] == 2.05)) )
        
        
    @unittest.skip
    def test_load(self):
        with open("points.dat") as f:
            data_mat = []
            for line in f:
                sline = line.split()
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat)
        
        means = mm.k_means(4,10)
        print(means)
        mm.plot_means(means)
        #print(mm)


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


       