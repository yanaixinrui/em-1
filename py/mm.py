#!/usr/bin/env python


"""
------------------------------------------------------------------------------
Classes to implement Gaussian mixture model with EM
------------------------------------------------------------------------------
"""
import numpy as np
import re

#import itertools # for product
#import sys        # for sys.float_info.max
#from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn import preprocessing

import unittest
import time
from scipy.stats import multivariate_normal
import random
from symbol import parameters

try:
    from scipy.misc  import logsumexp
except ImportError:
    def logsumexp(x,**kwargs):
        return np.log(np.sum(np.exp(x),**kwargs))


#============================================================================
class Mm():
    """ 
    """
    
    #--------------------------------------------------------------------------
    def __init__(self, point_list=[[]]): 
        self.X = np.array(point_list,copy = True, dtype=float) # X[i,point]
        self.mean_orig = self.X.mean(axis=0)
        self.var_orig = self.X.var(axis=0)
        self.X = preprocessing.scale(self.X)
        self.maxes = self.X.max(axis=0) # ndarray [d]
        self.mins = self.X.min(axis=0)  # ndarray [d]
        self.ranges = self.maxes - self.mins
        self.n,self.d = self.X.shape 
        #self.n_means = 1
        #self.means = np.mat(np.zeros((n_means, self.d),float))

        plt.figure(figsize=(6,6))
    
    #--------------------------------------------------------------------------   
    def __str__(self):
        s = "mm data: (N = " + str(self.n) + ", D = " + str(self.d) +")"
        for point in self.X:
            s += "\n  " + str(point)
        return s
    
    #--------------------------------------------------------------------------
    def k_means(self, k, n_iter=2):
        """ Lloyd's algorithm for solving k-means """
    
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
                    count_of_each_mean[k_i] = 1
                    point_sum[max_mean_i] = self.X[random.randrange(self.n)]
                    point_sum[max_mean_i][0] += random.random() - 0.5
                    
            #          [n_means x d] / [n_means]
            new_means = np.divide(point_sum, count_of_each_mean[:,np.newaxis]) 
            means = new_means
            self.plot_means(means)
        
        new_means *= np.sqrt(self.var_orig)
        new_means += self.mean_orig

        return new_means



    #--------------------------------------------------------------------------
    def em_init(self, k):
        """ Allocates large np.array data structures for em algorithm. 
            returns:
                parameters = [means, sigmas, pis]
                vars = [resp_kn, prob_masses, dists, counts]
        """

        # means  :[nm x d]
        means = np.random.rand(k,self.d) 
        means *= self.ranges  
        means += self.mins 

        # sigmas :[k][d x d]
        sigmas = np.empty(k,dtype=object)
        for k_i in range(k):
            sigmas[k_i] = 0.1 * np.eye(self.d, dtype=float)

        # pis    :[k]
        # (the mixing portion of each mean)
        pis = np.ones(k, dtype=float)*(1/k) # start w uniform distribution

        dists = np.empty(k,dtype=object)
        prob_masses = np.zeros((k,self.n),dtype=float)
        resp_kn = np.zeros((k,self.n),dtype=float)

        counts = np.zeros(k,dtype=float)
        parameters = [means, sigmas, pis]
        varz = [resp_kn, prob_masses, dists,counts]
        
        return parameters, varz

    #--------------------------------------------------------------------------
    def em_estep(self, parameters, varz):
        """ given parameters, calculate varz """
        
        means_kd, sigmas_kdd, pis_k          = parameters
        resp_kn, prob_masses, dists, counts  = varz
        k = len(pis_k)
        
        for k_i in range(k):
            dists[k_i] = multivariate_normal(means_kd[k_i],sigmas_kdd[k_i])
        
        for x_i,point in enumerate(self.X): # point is a [d] array
            # TODO: vectorize
            # calc prob masses & resp_kn
            for k_i in range(k):
                prob_mass = pis_k[k_i] * dists[k_i].pdf(point)
                prob_masses[k_i,x_i] = prob_mass
                                                
            # normalize responsibilities
            resp_kn[:,x_i] = prob_masses[:,x_i] / np.sum(prob_masses[:,x_i])
            assert(abs(np.sum(resp_kn[:,x_i]) -1) < 0.001)    
        
        # calc counts (Nk)
        for k_i in range(k):
            counts[k_i] = np.sum(resp_kn[k_i,:])

        return
    
    
    
    #--------------------------------------------------------------------------
    def em_mstep(self, parameters, varz):
        """ given varz, calculate parameters """
    
        means_kd, sigmas_kdd, pis_k          = parameters
        resp_kn, prob_masses, dists, counts  = varz
        k = len(pis_k)
        
        # calc means
        means_kd.fill(0.)
        #   calculate weighted sums of data points
        for x_i,point in enumerate(self.X): # point is a [d] array
            for k_i in range(k):
                means_kd[k_i] += resp_kn[k_i,x_i]*point
                
        #   average is sum/count
        for k_i in range(k):
            means_kd[k_i] /= counts[k_i]

        # calc pis
        total_counts = np.sum(counts)
        for k_i in range(k):
            pis_k[k_i] = counts[k_i]/total_counts
        assert(abs(np.sum(pis_k) -1) < 0.001)
        

        # calc sigmas
        for k_i in range(k):
            sigmas_kdd[k_i].fill(0.)

        for x_i,point in enumerate(self.X): # point is a [d] array
            # TODO: vectorize
            for k_i in range(k):
                d = point - means_kd[k_i]
                sigmas_kdd[k_i] += resp_kn[k_i,x_i] * np.outer(d,d)
                
        for k_i in range(k):
            sigmas_kdd[k_i] /= counts[k_i]                
            sigmas_kdd[k_i] += 0.000001 * np.eye(self.d) # prevent singularity

        return

    #--------------------------------------------------------------------------
    def em_for(self, k, n_iter=2):
        """ Gmm 
            k is the number of Gaussian mixtures, meaning we have k
            separate Gaussian distributions, each represented by its own
            mean and variance.
            The graphical models is to roll a k-sided die --> 
              then select a sample from the k'th distribution.
              
            probability is not represented in log form
            
         """
            
        #-----------------------------------------------------
        # INITIALIZE
        
        parameters, varz = self.em_init(k)

        means_kd, sigmas_kdd, pis_k          = parameters
        resp_kn, prob_masses, dists, counts  = varz
        
        self.plot_means(means_kd)
        
        #-----------------------------------------------------
        # ITERATION LOOP
        for i in range(n_iter):
            #------------------------
            # E-STEP
            #   given current params(mean & sigma),calc MLE cnts(responsibilites)

            self.em_estep(parameters,varz)            
                
            #------------------------
            # M-STEP
            #   given counts (resp_kn), update parameters (pis, means, sigmas)
            #   normalize prob_masses
            #   at this point, we want to find the relative probability of 
            #   of each k_means overall

            self.em_mstep(parameters, varz)
       
            self.plot_means(means_kd, sigmas_kdd)

        return means_kd, sigmas_kdd
        
    #--------------------------------------------------------------------------
    def plot_means(self, means, sigmas=[]):
        # sigmas=[[[.1,0],[0,.1]],[[.1,0],[0,.1]]]
        
        plt.clf()
        
        # plot data
        plt.scatter(self.X[:,0], self.X[:,1])
        
        # plot means
        plt.scatter(means[:,0], means[:,1], s=300,c='r')

        # plot sigmas
        if(sigmas != []):
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
                try:
                    plt.contour(x, y, rv.pdf(pos))
                except ValueError:
                    pass
                levels = [1, 2]
                #plt.contour(x, y, rv.pdf(pos), levels )
    
        plt.pause(0.001)

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
        means = mm.k_means(2)

        # make sure the correct means are discovered
        self.assertTrue((abs(means[0,0] - 1.05) < 0.01 and \
                         abs(means[0,1] - 1.05) < 0.01) or \
                       (abs(means[0,0] - 2.05) < 0.01 and \
                        abs(means[0,1] - 2.05) < 0.01) )
        self.assertTrue((abs(means[1,0] - 2.05) < 0.01 and \
                         abs(means[1,1] - 2.05) < 0.01) or \
                       (abs(means[1,0] - 1.05) < 0.01 and \
                        abs(means[1,1] - 1.05) < 0.01) )

    @unittest.skip
    def test_em_for_simple(self):
        print("\n...test_simple(...)")
        data_mat = np.mat('1 1; 2 2; 1.1 0.9; 2.1,2.0')
        mm = Mm(data_mat)
        #print(mm)
        
        means = mm.em_for(2, 20)
        #print(means)
        #mm.plot_means(means)
        
        '''
        self.assertTrue(((means[0,0] == 1.05) and (means[0,1] == 1.05)) or \
                   ((means[1,0] == 1.05) and (means[1,1] == 1.05)) )
        self.assertTrue(((means[0,0] == 2.05) and (means[0,1] == 2.05)) or \
                   ((means[1,0] == 2.05) and (means[1,1] == 2.05)) )
        '''
        
    @unittest.skip
    def test_k_means_load(self):
        with open("points.dat") as f:
            data_mat = []
            for line in f:
                sline = line.split()
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat)
        k = 4
        n_iter = 30
        means = mm.k_means(k, n_iter)
        print(means)
        mm.plot_means(means)

    #@unittest.skip
    def test_em_for(self):
        #with open("points.dat") as f:
        with open("decep.dat") as f:
            data_mat = []
            for line in f:
                #sline = line.split(', ')
                sline = re.findall(r'[^,;\s]+', line)
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat)
        k = 2
        n_iter = 50
        means, sigmas = mm.em_for(k, n_iter)
        print(means)
        mm.plot_means(means,sigmas)
        #pause = input('Press enter when complete: ')

    def test_em_for2(self):
        with open("points.dat") as f:
        #with open("decep.dat") as f:
            data_mat = []
            for line in f:
                #sline = line.split(', ')
                sline = re.findall(r'[^,;\s]+', line)
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat)
        for k in range(2,5):
            n_iter = 100
            means, sigmas = mm.em_for(k, n_iter)
            print(means)
            mm.plot_means(means,sigmas)
            pause = input('Press enter when complete: ')


    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class Mm complete..............\n")
        
#============================================================================
if __name__ == '__main__':
    #test_em_simple()
    try:
        unittest.main()
    except AssertionError:
        pass
    
# 11:00
# 2:40 


       