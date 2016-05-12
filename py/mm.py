#!/usr/bin/env python


"""
------------------------------------------------------------------------------
Classes to implement Gaussian mixture model with EM
------------------------------------------------------------------------------
"""
import numpy as np
import itertools # for product
import sys        # for sys.float_info.max
from collections import defaultdict
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

    def __init__(self, point_list=[]): 
        self.X = np.matrix(point_list,copy = True) # X[i,point]
        self.n,self.d = self.X.shape 
    
    def __str__(self):
        s = "mm data: (N = " + str(self.n) + ", D = " + str(self.d) +")"
        for point in self.X:
            s += "\n  " + str(point)
        return s
    

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
    
    def test_simple(self):
        load = 
        with open("points.dat") as f:
            for line in f:
                sline = line.split(" ")
                preamble = sline[0].split("_")
        
                if preamble[0] == 'T':
                    cur_tag_i  = self.t2i[preamble[1]]
                    next_tag_i = self.t2i[preamble[2]]
                    self.T[cur_tag_i,next_tag_i] = float(sline[1]) 
        
                elif preamble[0] == 'E':
                    tag_i  = self.t2i[preamble[1]]
                    word_i = self.w2i[preamble[2]]
                    self.E[word_i,tag_i] = float(sline[1]) 
                
                else:  
                    assert(True), "ERROR: unknown weight file entry" 
        m = Mm()

    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class Mm complete..............\n")
        
#============================================================================
if __name__ == '__main__':
    unittest.main()
       