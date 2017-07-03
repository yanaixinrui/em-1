#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Classes to implement mixture model clustering. Includes kmeans and Gaussian 
Mixture Model (MM) with Expectation Maximiz. (EM). Implementations are done in 
both 'for loop' as well as vectorized Numpy for profiling comparison.

To run individual unittests:
    $ python3 mm.py TestMm.test_k_means_simple

To profile:    
    $ python3 -O mm.py profile

    

-------------------------------------------------------------------------------
"""
import numpy as np
import re
import csv

import sys        # for sys.argv
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from scipy.stats import multivariate_normal
import random
import pandas as pd

try:
    from scipy.misc  import logsumexp
except ImportError:
    def logsumexp(x,**kwargs):
        return np.log(np.sum(np.exp(x),**kwargs))

import unittest
import time

#https://zameermanji.com/blog/2012/6/30/undocumented-cprofile-features/
import cProfile
import pstats 

#============================================================================
class Mm():
    """ Mixture Model class which implements both k-means and EM Gaussian
        mixture models.
    """
    
    #--------------------------------------------------------------------------
    def __init__(self, point_list=[[]],point_weights=[]): 
        """ Allocates data structures and standardizes input data. 
              New np array is allocated and data is copied from point_list.
              Data is standardized using sklearn.preprocessing.
            
        """
        self.MIN_VAR = 0.0001 # min variance for GMM (for numer. stability)
        self.X_nd = np.array(point_list, copy=True, dtype=float) # X[i,point]
        self.n,self.d = self.X_nd.shape 
        if point_weights == []:
            self.X_weights_n = np.ones(self.n)
        else:
            assert (len(point_weights) == self.n), \
                'len(point_weights) != len(point_list)' 
            self.X_weights_n = np.array(point_weights, copy=True, dtype=float)
            # normalize weights so that they sum to n
            self.X_weights_n *= self.n/sum(self.X_weights_n)

        self.mean_orig = self.X_nd.mean(axis=0)
        self.var_orig = self.X_nd.var(axis=0)
        self.maxes = self.X_nd.max(axis=0) # ndarray [d]
        self.mins = self.X_nd.min(axis=0)  # ndarray [d]
        self.ranges = self.maxes - self.mins        

        if __debug__ and plt:
                plt.figure(figsize=(6,6))
    
    #--------------------------------------------------------------------------   
    def __str__(self):
        s = "mm data: (N = " + str(self.n) + ", D = " + str(self.d) +")"
        for point in self.X_nd:
            s += "\n  " + str(point)
        return s
    
    #--------------------------------------------------------------------------
    def k_means(self, k, n_iter=2):
        """ Lloyd's algorithm for solving k-means (aka hard k-means)
            Code is not optimized. Note: you should standardize
            self.X_nd first if you don't want to minimize straight forward 
            Euclidian distance.
            
            returns: 
                means_kd # np.array
        """
    
        means_kd = np.random.rand(k,self.d) # [nm x d] mat
        means_kd = means_kd * self.ranges # [nm x d] * [d] 
        means_kd = means_kd + self.mins # [nm x d] + [d]
        
        self.plot_means(means_kd)

        for i in range(n_iter):
            # Given the current means, calculate counts_k for each
  
            count_of_each_mean_k = np.zeros(k, dtype=float)
            point_sum_kd = np.zeros((k,self.d), dtype=float)
            
            # TODO: vectorize this for loop
            for i,point_d in enumerate(self.X_nd): # point is a row [1 x 2]
                # the "probability mass" is just inverse distance sq
                difference_kd =  (means_kd - point_d)          # [nm x d] - [1 x d]
                dist_sq_k = np.sum((difference_kd * difference_kd), axis=1)
                prob_mass_k = np.reciprocal(dist_sq_k)  # [2 x 1]
                max_mean_j = np.argmax(prob_mass_k)
                count_of_each_mean_k[max_mean_j] += self.X_weights_n[i] 
                point_sum_kd[max_mean_j] += point_d*self.X_weights_n[i]
                    
            # check for zeros
            for  k_i in range(k):
                if(count_of_each_mean_k[k_i] == 0):
                    # randomly assign a point (with a little noise) from data 
                    # set to this mean
                    point_i = random.randrange(self.n)
                    count_of_each_mean_k[k_i] = self.X_weights_n[point_i]
                    point_sum_kd[k_i] = self.X_nd[point_i]
                    point_sum_kd[k_i][0] += random.random() - 0.5
                    
            #          [n_means x d] / [n_means]
            new_means_kd = np.divide(point_sum_kd, 
                                     count_of_each_mean_k[:,np.newaxis])
            means_kd = new_means_kd
            self.plot_means(means_kd)
        
        #new_means_kd *= np.sqrt(self.var_orig)
        #new_means_kd += self.mean_orig

        return new_means_kd

    #--------------------------------------------------------------------------
    def em_init(self, k):
        """ Allocates large np.array data structures for em algorithm. 
        
            returns:
                parameters = [means_kd, sigmas_k, pis_k]
                varz = [resp_kn,        # the responsibility of each Gaussian
                        prob_masses_kn, # each point's prob. for each Gaussian
                        counts_k]       # the num of points in each Gaussian
        """

        # means  :[nm x d]
        means_kd = np.random.rand(k,self.d) 
        means_kd *= self.ranges  
        means_kd += self.mins 

        # sigmas :[k][d x d]
        sigmas_k = np.empty(k,dtype=object)
        for k_i in range(k):
            #sigmas_k[k_i] = 0.1 * np.eye(self.d, dtype=float)
            sigmas_k[k_i] = self.var_orig * np.eye(self.d, dtype=float) + \
                            0.1 * np.eye(self.d, dtype=float)

        # pis    :[k]
        # (the mixing portion of each mean)
        pis_k = np.ones(k, dtype=float)*(1/k) # start w uniform distribution

        prob_masses_kn = np.zeros((k,self.n),dtype=float)
        resp_kn = np.zeros((k,self.n),dtype=float)

        counts_k = np.zeros(k,dtype=float)
        parameters = [means_kd, sigmas_k, pis_k]
        varz = [resp_kn, prob_masses_kn, counts_k]
        
        return parameters, varz

    #--------------------------------------------------------------------------
    def em_estep(self, parameters, varz):
        """ Non-vectorized vesion - given parameters, calculates varz """
        
        means_kd, sigmas_kdd, pis_k          = parameters
        resp_kn, prob_masses_kn, counts_k  = varz
        k = len(pis_k)
        dists_k = np.empty(k,dtype=object)
        
        for k_i in range(k):
            dists_k[k_i] = multivariate_normal(means_kd[k_i],sigmas_kdd[k_i])
        
        for x_i,point in enumerate(self.X_nd): # point is a [d] array
            # calc prob masses & resp_kn
            for k_i in range(k):
                prob_mass = pis_k[k_i] * dists_k[k_i].pdf(point) * self.X_weights_n[x_i]
                prob_masses_kn[k_i,x_i] = prob_mass
                                                
            # normalize responsibilities
            resp_kn[:,x_i] = prob_masses_kn[:,x_i] / np.sum(prob_masses_kn[:,x_i])
            assert(abs(np.sum(resp_kn[:,x_i]) -1) < 0.001)    
        

    
    #--------------------------------------------------------------------------
    def em_estep_v(self, parameters, varz):
        """ Vectorized version - given parameters, calculates varz """
        
        means_kd, sigmas_kdd, pis_k            = parameters
        resp_kn, prob_masses_kn, counts_k  = varz
        k = len(pis_k)
        dists_k = np.empty(k,dtype=object)
                
        for k_i in range(k):
            dists_k[k_i] = multivariate_normal(means_kd[k_i],sigmas_kdd[k_i])
        
        '''
        for x_i,point in enumerate(self.X_nd): # point is a [d] array
            # TODO: vectorize
            # calc prob masses & resp_kn
            for k_i in range(k):
                prob_mass = pis_k[k_i] * dists[k_i].pdf(point)
                prob_masses_kn[k_i,x_i] = prob_mass
                                                
            # normalize responsibilities
            resp_kn[:,x_i] = prob_masses_kn[:,x_i] / np.sum(prob_masses_kn[:,x_i])
            assert(abs(np.sum(resp_kn[:,x_i]) -1) < 0.001)    
        '''
            
        # it is doubtful that vectorizing this will help
        for k_i in range(k):
            # calc prob masses & resp_kn
            prob_masses_kn[k_i,:] = pis_k[k_i] * \
                                    dists_k[k_i].pdf(self.X_nd) * \
                                    self.X_weights_n
                                                
        '''
        for x_i,point in enumerate(self.X_nd): # point is a [d] array
            # normalize responsibilities
            resp_kn[:,x_i] = prob_masses_kn[:,x_i] / np.sum(prob_masses_kn[:,x_i])
            assert(abs(np.sum(resp_kn[:,x_i]) -1) < 0.001)    
        '''
        # normalize responsibilities
        resp_kn[:,:] = np.divide(prob_masses_kn, 
                                 np.sum(prob_masses_kn, axis=0)[np.newaxis,:])
        # TODO: create a vectorized assert
        #assert(abs(np.sum(resp_kn[:,x_i]) -1) < 0.001)   
         


    #--------------------------------------------------------------------------
    def em_mstep(self, parameters, varz):
        """ Non-vectorized - given varz, calculates parameters """
    
        means_kd, sigmas_kdd, pis_k          = parameters
        resp_kn, prob_masses_kn, counts_k  = varz
        k = len(pis_k)
        
        # calc counts_k (Nk)
        for k_i in range(k):
            counts_k[k_i] = np.sum(resp_kn[k_i,:])
        
        # calc means
        means_kd.fill(0.)
        #   calculate weighted sums of data points
        for x_i,point in enumerate(self.X_nd): # point is a [d] array
            for k_i in range(k):
                means_kd[k_i] += resp_kn[k_i,x_i]*point
                
        #   average is sum/count
        for k_i in range(k):
            means_kd[k_i] /= counts_k[k_i]

        # calc pis           
        for k_i in range(k):
            pis_k[k_i] = counts_k[k_i]/self.n
        assert(abs(np.sum(pis_k) -1) < 0.001)
        
        # calc sigmas
        for k_i in range(k):
            sigmas_kdd[k_i].fill(0.)

        for x_i,point in enumerate(self.X_nd): # point is a [d] array
            for k_i in range(k):
                d = point - means_kd[k_i]
                sigmas_kdd[k_i] += resp_kn[k_i,x_i] * np.outer(d,d)
                
        for k_i in range(k):
            sigmas_kdd[k_i] /= counts_k[k_i]                
            sigmas_kdd[k_i] += self.MIN_VAR * np.eye(self.d) # prevent singularity

    #--------------------------------------------------------------------------
    def em_mstep_v(self, parameters, varz):
        """ Vectorized - given varz, calculate parameters """
    
        means_kd, sigmas_kdd, pis_k          = parameters
        resp_kn, prob_masses_kn, counts_k  = varz
        k = len(pis_k)
        
        # calc counts_k (Nk)
        #for k_i in range(k):
        #    counts_k[k_i] = np.sum(resp_kn[k_i,:])
        counts_k[:] = np.sum(resp_kn, axis=1)        
    
        # calc means
        means_kd.fill(0.)
        #   calculate weighted sums of data points
        #for x_i,point in enumerate(self.X_nd): # point is a [d] array
            #for k_i in range(k):
            #    means_kd[k_i] += resp_kn[k_i,x_i]*point
        means_kd[:] =  np.einsum('kn,nd->kd',resp_kn, self.X_nd) 
        
        #   average is sum/count
        #for k_i in range(k):
        #    means_kd[k_i] /= counts[k_i]
        means_kd /= counts_k[:,np.newaxis]

        # calc pis
        #for k_i in range(k):
        #    pis_k[k_i] = counts_k[k_i]/self.n
        pis_k[:] = counts_k/self.n
        assert(abs(np.sum(pis_k) -1) < 0.001)
        

        # calc sigmas
        for k_i in range(k):
            sigmas_kdd[k_i].fill(0.)

        '''
        for x_i,point in enumerate(self.X_nd): # point is a [d] array
            # TODO: vectorize
            for k_i in range(k):
                d = point - means_kd[k_i]
                sigmas_kdd[k_i] += resp_kn[k_i,x_i] * np.outer(d,d)
        '''
        #d_knd = self.X_nd[np.newaxis,:,:] - means_kd[:,np.newaxis,:]
        #d_knd *= d_knd
        for k_i in range(k):
            #for x_i,point in enumerate(self.X_nd): # point is a [d] array
            diff_nd = np.sqrt(resp_kn[k_i,:,np.newaxis]) * (self.X_nd - means_kd[k_i])
            sigmas_kdd[k_i] = np.sum(np.einsum('nd,nD->ndD', diff_nd, diff_nd),axis=0)
               
        for k_i in range(k):
            sigmas_kdd[k_i] /= counts_k[k_i]                
            sigmas_kdd[k_i] += self.MIN_VAR * np.eye(self.d) # prevent singularity

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
        resp_kn, prob_masses_kn, counts_k  = varz
        
        if __debug__:
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
            #   given counts_k (resp_kn), update parameters (pis, means, sigmas)
            #   normalize prob_masses_kn
            #   at this point, we want to find the relative probability of 
            #   of each k_means overall

            self.em_mstep(parameters, varz)
       
            if __debug__:
                self.plot_means(means_kd, sigmas_kdd)

        return means_kd, sigmas_kdd
        
    #--------------------------------------------------------------------------
    def em_v(self, k, n_iter=2):
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

        means_kd, sigmas_kdd, pis_k   = parameters
        resp_kn, prob_masses_kn, counts_k  = varz
        
        if __debug__:
            self.plot_means(means_kd)
        
        #-----------------------------------------------------
        # ITERATION LOOP
        for i in range(n_iter):
            #------------------------
            # E-STEP
            #   given current params(mean & sigma),calc MLE cnts(responsibilites)

            self.em_estep_v(parameters,varz)            
                
            #------------------------
            # M-STEP
            #   given counts_k (resp_kn), update parameters (pis, means, sigmas)
            #   normalize prob_masses_kn
            #   at this point, we want to find the relative probability of 
            #   of each k_means overall

            self.em_mstep_v(parameters, varz)
       
            if __debug__:
                self.plot_means(means_kd, sigmas_kdd)

        return means_kd, sigmas_kdd
                
    #--------------------------------------------------------------------------
    def em_gpu(self, k, n_iter=2):
        """ gnumpy based implementation - Gmm 
            k is the number of Gaussian mixtures, meaning we have k
            separate Gaussian distributions, each represented by its own
            mean and variance.
            The graphical models is to roll a k-sided die --> 
              then select a sample from the k'th distribution.
              
            probability is not represented in log form
            
        """

        """
        TODO
        """
        pass
                         
    #--------------------------------------------------------------------------
    def plot_means(self, means, sigmas=np.array([]), tags=[]):
        """ Plots the datapoints along with topographic map of means
            and sigmas.  Works only for 2D data.
            tags is used to specify different classes for the datapoints 
            so the they are colored differently.
            
            example format:
              sigmas=[[[.1,0],[0,.1]],[[.1,0],[0,.1]]]
              tags = ['1','0','1','1','0'] # length n 
        """
        plt.clf()
        
        # If we got a tags arg, color points green for '1', red for '0'
        # otherwise color the points blank (white)
        if tags:
            colors = ['g' if x == '1' else 'r' for x in tags]
        else:
            colors = 'k'
        plt.scatter(self.X_nd[:,0], self.X_nd[:,1], c=colors)
        
        # plot means
        plt.scatter(means[:,0], means[:,1], s=300,c='r')

        # plot sigmas
        if sigmas.size != 0:
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

                try:
                    #plt.contour(x, y, rv.pdf(pos),cmap=cm.coolwarm)
                    plt.contour(x, y, rv.pdf(pos))
                except ValueError:
                    pass
    
        plt.pause(0.001)

    #--------------------------------------------------------------------------
    def calc_mse(self, means, sigmas):
        """ given a set of means and sigmas
              for each point (in self.X_nd)
                find the closest mean
                add the sq distance to mse
            divide by N to get mean

            returns the total MSE
        """
        # TODO        
        pass
    
    #--------------------------------------------------------------------------
    def calc_P(self, means, sigmas):
        """ returns the probability of the data """
        # TODO        
        pass

    #--------------------------------------------------------------------------
    def cluster(self, infile='example/interrogator.csv', 
                outfile='example/clusters.csv', 
                features=[' AU06_r',' AU12_r'], 
                k=5):
        """ Loads data from infile, runs GMM, writes clusters to outfile.
        
        """
        print("\n...clustering(...)")
        df = pd.read_csv(infile) 
        
        if __debug__:
            plt.close()
        self.__init__(df.loc[:,features])
        n_iter = 200
        means, sigmas = mm.em_v(k, n_iter)
        print(means)
        if __debug__:
            mm.plot_means(means,sigmas)
            plt.title('GMM results')
        cluster_data = np.concatenate((means,sigmas[:,np.newaxis]),axis=1)
        df_clusters = pd.DataFrame(data=cluster_data,columns=features+['sigmas'])
        df_clusters.to_csv(outfile,index=False)
        if __debug__:
            plt.savefig(cluster_outfile + '.png')

    
#============================================================================
class TestMm(unittest.TestCase):
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
        mm = Mm(a)
        print(mm)
    
    #@unittest.skip
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
        
    #@unittest.skip
    def test_k_means(self):
        print("\n...test_k_means(...)")
        with open("points.dat") as f:
            data_mat = []
            for line in f:
                sline = line.split()
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat,np.random.rand(len(data_mat)))
        k = 4
        n_iter = 30
        means = mm.k_means(k, n_iter)
        print(means)
        mm.plot_means(means)
        plt.title('test_k_means')


    #@unittest.skip
    def test_em_for_simple(self):
        print("\n...test_em_for_simple(...)")
        data_mat = np.mat('1 1; 2 2; 1.1 0.9; 2.1,2.0')
        mm = Mm(data_mat,np.random.rand(len(data_mat)))
        #print(mm)
        
        means, sigmas = mm.em_for(2, 20)
        print(means)
        mm.plot_means(means)
        plt.title('test_em_for_simple')
        
        self.assertTrue((abs(means[0,0] - 1.05) < 0.01 and \
                         abs(means[0,1] - 0.95) < 0.01) or \
                       (abs(means[0,0] - 2.05) < 0.01 and \
                        abs(means[0,1] - 2.00) < 0.01) )
        self.assertTrue((abs(means[1,0] - 2.05) < 0.01 and \
                         abs(means[1,1] - 2.00) < 0.01) or \
                       (abs(means[1,0] - 1.05) < 0.01 and \
                        abs(means[1,1] - 0.95) < 0.01) )

    #@unittest.skip
    def test_em_for(self):
        print("\n...test_em_for(...)")
        with open("points.dat") as f:
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
        plt.title('test_em_for')
        #pause = input('Press enter when complete: ')

    #@unittest.skip    
    def test_em_v(self):
        print("\n...test_em_v(...)")
        with open("points.dat") as f:
            data_mat = []
            for line in f:
                #sline = line.split(', ')
                sline = re.findall(r'[^,;\s]+', line)
                assert(len(sline) == 2)
                data_mat.append(sline)
        mm = Mm(data_mat)
        k = 2
        n_iter = 50
        means, sigmas = mm.em_v(k, n_iter)
        print(means)
        mm.plot_means(means,sigmas)
        plt.title('test_em_v')        
        #pause = input('Press enter when complete: ')

    #@unittest.skip
    def test_em_for2(self):
        print("\n...test_em_for2(...)")
        with open("points.dat") as f:
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
            plt.title('test_em_for2')
            #pause = input('Press enter when complete: ')


    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class Mm complete..............\n")
        
#============================================================================
class ProfileMm(unittest.TestCase):
    """ Profile the alternative methods of Mm """

    def compare_all(self):
        self.do_loop('test.dat',range(4,5),30)
    
    def do_loop(self, data_file, k_range, n_iter):
        with open(data_file) as f:
            data_mat = []
            for line in f:
                sline = re.findall(r'[^,;\s]+', line)
                #assert(len(sline) == 2)  # eventually remove
                data_mat.append(sline)
        mm = Mm(data_mat)
        k = 4 # TODO range
        #cProfile.run('means, sigmas = mm.em_v(k, n_iter)')
        cProfile.runctx('mm.em_v(k, n_iter)', globals(), locals(),'em_v_stats')
        cProfile.runctx('mm.em_for(k, n_iter)', globals(), locals(),'em_for_stats')
        
        stats = pstats.Stats('em_v_stats')
        stats.strip_dirs()
        stats.sort_stats('filename')
        stats.print_stats()

        stats = pstats.Stats('em_for_stats')
        stats.strip_dirs()
        stats.sort_stats('filename')
        stats.print_stats()
    
    def generate_data_file(self, data_file, k, d, n):
        """ Generates test data file for use in testing.
                n samples
                k number of means used
                d dimensional
            Means and covariances are randomly sampled over the unit hypercube.
        """
        
        # means  :[k x d]
        means_kd = np.random.rand(k,d) 

        # sigmas :[k][d x d]
        # must make sure that the matrix is symetric and
        # the nondiagonal components Sij are <=  Sii and Sjj
        sigmas_kdd = np.empty(k,dtype=object)
        dists_k = np.empty(k,dtype=object)
        for k_i in range(k):
            sigmas_kdd[k_i] = 0.01 * np.eye(d,dtype=float)
            dists_k[k_i] = multivariate_normal(means_kd[k_i], sigmas_kdd[k_i])
        
        
        # (the mixing portion of each mean)
        pis = np.random.rand(k)
        pis /= pis.sum()  # normalize so sum is 1

        counts_k = np.zeros(k,dtype=int)
        for k_i in range(k-1):
            counts_k[k_i] = int(n*pis[k_i])
        counts_k[k-1] = n - counts_k.sum() 
        
        # generate data
        X_nd = np.zeros((n,d), dtype=float)
        index = 0
        for k_i in range(k):    
            X_nd[index:index+counts_k[k_i],:] = dists_k[k_i].rvs(counts_k[k_i])
            index += counts_k[k_i]
        
        # write data to file
        with open(data_file, 'w') as f:
            for point in X_nd:
                f.write(str(point)[1:-1]) # strip brackets
                f.write('\n')
        
        print('data generation complete')
        
        
#============================================================================
def process_decep_data(fname='m_out_abs.csv'):
    """ TODO:
          currently the code loads decep.dat, then runs clustering with K=2
           
          we need to modify the code to:
             1. instead load our csv file
               for each pair of colums:
               2. grab subdata
               3. run clustering on it
               4. run the evaluation code (mse) on the result
               5. write the results (mu, sigma, and mse) to a file

    """
    print('processing deception data')

    f = open(fname, 'rt')
    data = []
    try:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(np.array(row))
            #data.append(row)
    finally:
        f.close()
    
    data = np.array(data) # data[filenum, datafield] # all data is strings, even numbers
    print(data.shape)
    print(header)
    
    IsTrue = np.array(data[:,2], float)
    print('(n_files,n_fields)=', data.shape)        
    
    data_mat = data[1:,[3,7]] # we only want ResponseTime and joy columns
    is_true = data[1:,2]      # truth or bluff?
    '''
    with open(fname) as f:
        data_mat = []
        for line in f:
            #sline = line.split(', ')
            sline = re.findall(r'[^,;\s]+', line)
            assert(len(sline) == 2)
            data_mat.append(sline)
    '''
    mm = Mm(data_mat)
    k = 2
    n_iter = 50
    means, sigmas = mm.em_v(k, n_iter)
    
    # TODO, try out code below
    #mm.calc_mse(means, sigmas)

    print(means)
    mm.plot_means(means,sigmas,is_true)    
    
#============================================================================
def voice():
    with open("voice.dat") as f:
        data_mat = []
        for line in f:
            sline = line.split()
            #sline = re.findall(r'[^,;\s]+', line)
            assert(len(sline) == 2)
            data_mat.append(sline)
    mm = Mm(data_mat)
    k = 2
    n_iter = 50
    means, sigmas = mm.em_v(k, n_iter)
    print(means)
    mm.plot_means(means,sigmas)

    plt.figure()
    k = 3
    n_iter = 50
    means, sigmas = mm.em_v(k, n_iter)
    print(means)
    mm.plot_means(means,sigmas)
    
    pause = input('Press enter when complete: ')

#============================================================================
def voice2():
    from itertools import combinations
    USE_POLY = False
    
    f = open("vdata_with_sent4.csv", 'rt')
    data = []
    try:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(np.array(row))
    finally:
        f.close()
        
    header = np.array(header)
    data = np.array(data) # data[patient, datafield] # all data is strings, even numbers
    print(data.shape)
    visit_rating_questions =  np.array([387,388,389,391,392,610]) # problem with 390 - empty fields
    
    # reverse polarity of question 392 data
    for i in range(data.shape[0]):
        new_val = 6 - int(data[i,392])
        data[i,392] = str(new_val)
    
    #key_stats = np.array([575, 577, 573,564,576, 566,578,588,589,590,591,592, \
    #                      600,601,602,603,604,605,606,607])
    key_stats = np.array([575, 577, 573,588,589,590,591,592, \
                          600,601,602,603,604,605,606,607])
    key_stats_data = np.array(data[:,key_stats],float)
    
    print(header[key_stats])
    
    features = key_stats_data
    results = []
    good_results = []
    #for num_features in range(1,5):
    for num_features in range(2,3):
        feature_combo_list = list(combinations(range(14),num_features))
    
        for q in [visit_rating_questions[-1]]:
            Ratings = np.array(data[:,q],int)
            min_val = min(Ratings)            
            y_n = (Ratings == min_val)
            colors = ['g' if y==1 else 'r' for y in y_n]
            for features in feature_combo_list:
                X_nd = key_stats_data[:,features]
                if(USE_POLY):
                    X_nd = poly.fit_transform(X_nd)
                X_nd = key_stats_data[:,features]
                gmm = Mm(X_nd)
                k = 2
                n_iter = 50
                print('FEATURES:',str(header[key_stats[list(features)]]))
                means, sigmas = gmm.em_v(k, n_iter)
                print(means)
                gmm.plot_means(means,sigmas)    
    
    
#============================================================================
if __name__ == '__main__':

    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'generate_test_data'):
            profiler = ProfileMm()
            profiler.generate_data_file('test.dat', 4, 3, 50000) # k d n
        elif (sys.argv[1] == 'profile'):
            print('Profiling starting...')
            profiler = ProfileMm()
            profiler.compare_all()
        elif (sys.argv[1] == 'deception'):
            # TODO
            data_file = 'm_out_abs.csv' # change to your datafile
            process_decep_data(data_file) 
        elif (sys.argv[1] == 'voice'):
            voice()
        elif (sys.argv[1] == 'voice2'):
            voice2()
        elif (sys.argv[1] == 'cluster'):
            mm = Mm()
            mm.cluster()
        else:
            unittest.main()
    else:
        unittest.main()
        
