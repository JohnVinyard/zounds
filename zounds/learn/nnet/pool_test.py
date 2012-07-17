import unittest
import numpy as np
from model.pipeline import Pipeline
from learn.nnet.nnet import NeuralNetwork
from learn.fetch import Fetch
from learn.preprocess import NoOp

class MockNN(NeuralNetwork):
    
    def __init__(self,indim,hdim):
        NeuralNetwork.__init__(self)
        self._indim = indim
        self._hdim = hdim
    
    @property
    def indim(self):
        return self._indim
    
    @property
    def hdim(self):
        return self._hdim
    
    @property
    def dim(self):
        return self._hdim
    
    def activate(self,inp):
        '''
        reconstruct input
        '''
        pass
    
    def __call__(self,data):
        '''
        extract features
        '''
        ld = len(data)
        return data.sum(1).repeat(self._hdim).reshape((ld,self._hdim))
        

class PoolTests(unittest.TestCase):
    
    
    def setUp(self):
        self.to_remove = []
    
    def tearDown(self):
        for tr in self.to_remove:
            del Pipeline[tr]
    
    