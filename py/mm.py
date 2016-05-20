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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import unittest
import time
from scipy.stats import multivariate_normal
import random

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
        self.X = np.array(point_list,copy = True, dtype=float) # X[i,point]
        self.n,self.d = self.X.shape 
        #self.n_means = 1
        #self.means = np.mat(np.zeros((n_means, self.d),float))
        self.maxes = self.X.max(axis=0) # ndarray [d]
        self.mins = self.X.min(axis=0)  # ndarray [d]
        self.ranges = self.maxes - self.mins
        plt.figure(figsize=(6,6))
        
    def __str__(self):
        s = "mm data: (N = " + str(self.n) + ", D = " + str(self.d) +")"
        for point in self.X:
            s += "\n  " + str(point)
        return s
    
    def k_means(self, k, n_iter=2):
        """ Lloyd's algorithm for solving k-means """
        
        # TODO: handle zero counts
        
        
        # Q: determine whether numpy compiles below statements into single cmd?
        means = np.random.rand(k,self.d) # [nm x d] mat
        means = means * self.ranges # [nm x d] * [d] 
        means = means + self.mins # [nm x d] + [d]
        
        self.plot_means(means)

        for i in range(n_iter):
            # given the current means, calculate counts for each
            count_of_each_mean = np.zeros(k, dtype=float)
            point_sum = np.zeros((k,self.d), dtype=float)
            
            # TODO: vectorize this for loop
            for point in self.X: # point is a row [1 x 2]
                # the "probability mass" is just inverse distance sq
                difference =  (means - point)          # [nm x d] - [1 x d]
                dist_sq = np.sum((difference * difference), axis=1)
                prob_mass = np.reciprocal(dist_sq)  # [2 x 1]
                max_mean_i = np.argmax(prob_mass)
                count_of_each_mean[max_mean_i] += 1 
                point_sum[max_mean_i] += point
                    
            # check for zeros
            for  k_i in range(k):
                if(count_of_each_mean[k_i] == 0):
                    # randomly assign a point (with a little noise) from data 
                    # set to this mean
                    count_of_each_mean[k_i] == 1
                    point_sum[max_mean_i] = self.X[random.randrange(self.n)]
                    point_sum[max_mean_i][0] += random.random() - 0.5
                    
            #          [n_means x d] / [n_means]
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
              then select a sample from the k'th distribution.
              
            probability is not represented in log form
            
         """
            
        # initialize means with k_means
        means = self.k_means(k,5)
        sigmas = []
        # pi is the prob of selecting each mean
        pis = np.ones(k, dtype=float)*(1/k) # start w uniform distribution

        dists = np.empty(k,dtype=object)
        sigmas = np.empty(k,dtype=object)
        prob_masses = np.zeros((k,self.n),dtype=float)
        responsibilities = np.zeros((k,self.n),dtype=float)

        for k_i in range(k):
            #sigmas[k_i] = np.zeros((self.d,self.d), dtype=float)
            sigmas[k_i] = np.eye(self.d, dtype=float)
        for x_i,point in enumerate(self.X): # point is a [d] array
            # TODO: vectorize
            for k_i in range(k):
                d = point - means[k_i]
                #sigmas[k_i] += np.outer(d,d)
                sigmas[k_i][0,0] += d[0] * d[0]
                sigmas[k_i][1,1] += d[1] * d[1]
        self.plot_means(means,sigmas)

        for i in range(n_iter):
            # E-STEP
            # given the current parameters, calc MLE counts (prob mass) 
            # for each mean
            prob_mass_of_each_k = np.zeros(k, dtype=float)
            point_mass_of_each_k = np.zeros((k,self.d), dtype=float)
            for k_i in range(k):
                dists[k_i] = multivariate_normal(means[k_i], sigmas[k_i])
            
            for x_i,point in enumerate(self.X): # point is a [d] array
                # TODO: vectorize
                for k_i in range(k):
                    prob_mass = pis[k_i] * dists[k_i].pdf(point)
                    prob_mass_of_each_k[k_i] += prob_mass 
                    prob_masses[k_i,x_i] = prob_mass
                    point_mass_of_each_k[k_i] += prob_mass * point
                
                responsibilities[:,x_i] = prob_masses[:,x_i] / np.sum(prob_masses[:,x_i]) 
                    
            # M-STEP
            # given counts (prob masses), update parameters (pis, means, sigmas)
            # normalize prob_masses
            # at this point, we want to find the relative probability of 
            # of each k_means overall
            # SUM(point_mass_of_each_k) should be 1
            sum_of_prob_masses = np.sum(prob_mass_of_each_k)
            pis_new = prob_mass_of_each_k / sum_of_prob_masses
            #assert(abs(np.sum(pis_new) -1) < 0.001)
            
            # TODO: don't keep reallocating memory
            means_new = np.zeros_like(means)
            
            # TODO: vectorize
            for k_i in range(k):
                means_new[k_i] = point_mass_of_each_k[k_i] / prob_mass_of_each_k[k_i]

            # TODO: update sigmas
            for k_i in range(k):
                sigmas[k_i].fill(0.)
            #sigmas_new = np.zeros_like(sigmas)
            for x_i,point in enumerate(self.X): # point is a [d] array
                # TODO: vectorize
                for k_i in range(k):
                    d = point - means_new[k_i]
                    #sigmas_new[k_i] += responsibilities[k_i,x_i] * np.outer(d,d)
                    sigmas[k_i][0,0] += responsibilities[k_i,x_i] * d[0] * d[0]
                    sigmas[k_i][1,1] += responsibilities[k_i,x_i] * d[1] * d[1]
            for k_i in range(k):
                #sigmas[k_i] = sigmas[k_i] / prob_mass_of_each_k[k_i]
                sigmas[k_i] = sigmas[k_i] / np.sum(responsibilities[k_i,:])
                sigmas[k_i] += 0.000001 * np.eye(self.d) # prevent singularity
            pis = pis_new
            means = means_new
            sigmas = sigmas 
            
            #print(means)
            self.plot_means(means, sigmas)

        return means_new
        
    def plot_means(self, means, sigmas=[[[.1,0],[0,.1]],[[.1,0],[0,.1]]]):
        plt.clf()
        
        # plot data
        plt.scatter(self.X[:,0], self.X[:,1])
        # plot means
        plt.scatter(means[:,0], means[:,1], s=300,c='r')

        # plot sigmas
        for k_i in range(len(sigmas)):
            x_vals = np.linspace(self.mins[0], self.maxes[0], 50)
            y_vals = np.linspace(self.mins[1], self.maxes[1], 50)
            x, y = np.meshgrid(x_vals, y_vals)
            #x, y = np.mgrid[self.mins[0]:self.maxes[0]:.01, \
            #                self.mins[1]:self.maxes[1]:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x; pos[:, :, 1] = y
            #rv = multivariate_normal([0.0, 0.0], [[.1, .07], [0.07, .1]])
            rv = multivariate_normal(means[k_i], sigmas[k_i])
            #plt.contour(x, y, rv.pdf(pos),cmap=cm.coolwarm)
            plt.contour(x, y, rv.pdf(pos))
            levels = [1, 2]
            #plt.contour(x, y, rv.pdf(pos), levels )
    
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

    #@unittest.skip
    def test_em_simple(self):
        print("\n...test_simple(...)")
        data_mat = np.mat('1 1; 2 2; 1.1 0.9; 2.1,2.0')
        mm = Mm(data_mat)
        #print(mm)
        
        means = mm.em(2, 20)
        #print(means)
        #mm.plot_means(means)
        
        '''
        self.assertTrue(((means[0,0] == 1.05) and (means[0,1] == 1.05)) or \
                   ((means[1,0] == 1.05) and (means[1,1] == 1.05)) )
        self.assertTrue(((means[0,0] == 2.05) and (means[0,1] == 2.05)) or \
                   ((means[1,0] == 2.05) and (means[1,1] == 2.05)) )
        '''
        
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
    #test_em_simple()
    unittest.main()
    
    
# 11:00
# 2:40 


       