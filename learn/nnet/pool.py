from __future__ import division
import numpy as np

from nnet import NeuralNetwork
from learn.learn import Learn



class Pooling(NeuralNetwork,Learn):
    
    def __init__(self,layer0,layer1,pooling_method = np.max):
        self._layer0 = layer0
        self._layer1 = layer1
        self._pooling_method = pooling_method
    
    
    @property
    def indim(self):
        return self._layer1.indim
    
    @property
    def hdim(self):
        return self._layer1.hdim
    
    @property
    def dim(self):
        return self.hdim
    
    def _aggregate(self,data):
        return self._pooling_method(data,axis = 0)
    
    def _pool(self,data):
        ld = len(data) if len(data.shape) > 1 else 1
        # how many blocks are we pooling?
        # KLUDGE: I shouldn't know about the "learn" property here.  Pipeline
        # should probably have indim and outdim properties
        if 1 == ld:
            pool_size = data.shape[0] / self._layer0.learn.indim
        else:
            pool_size = data.shape[1] / self._layer0.learn.indim
        # reshape the data so that it can be processed by the layer 0 network
        data = data.reshape((pool_size * ld,self._layer0.learn.indim))
        # extract features with the layer 0 network
        l0_features = np.zeros((len(data),self._layer0.learn.hdim))
        
        # KLUDGE: Do this in "mini-batches" to lessen time spent in the
        # python for-loop
        for i in xrange(len(l0_features)):
            l0_features[i] = self._layer0(data[i])
        
        # initialize an array to hold the pooled features
        samples = np.zeros((ld,self.indim))
        for i in xrange(ld):
            start = i * pool_size
            stop = start + pool_size
            samples[i] = self._aggregate(l0_features[start : stop])
        
        return samples
            
    
    def train(self,samples,stopping_condition):
        pooled = self._pool(samples)
        self._layer1.train(pooled,stopping_condition)
    
    def __call__(self,data):
        pooled = self._pool(data)
        return self._layer1(pooled)
    
    def activate(self,data):
        pooled = self._pool(data)
        return self._layer1.activate(pooled)
        