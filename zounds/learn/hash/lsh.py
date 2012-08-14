import struct

import numpy as np
from scipy.cluster.vq import whiten,kmeans
from scipy.spatial.distance import cdist
from bitarray import bitarray

from zounds.learn.learn import Learn
from zounds.nputil import pack

class Lsh(Learn):
    
    def __init__(self,size,nhashes = 32):
        Learn.__init__(self)
        self.size = size
        self.nhashes = nhashes
        self.hashes = None
    
    def _check_unique(self,nsamples,hashes,samples):
        self.hashes = hashes.reshape((len(hashes),np.product(hashes.shape[1:])))
        samples = samples.reshape((len(samples),np.product(samples.shape[1:])))
        results = self(samples)
        
        s = set()
        [s.add(tuple(r)) for r in results]
        return len(s)
        
    
#    def train(self,data,stopping_condition):
#        '''
#        Assume centered data that is in a completely randomized order.
#        Look for hash functions (example data) that approximately splits
#        randomly selected data in half.
#        '''
#        nsamples = 2000
#        best_score = 0
#        best = None
#        l = len(data) - 1
#        for i in range(100):
#            self.hashes = data[np.random.random_integers(0,l,self.nhashes)]
#            print self.hashes.shape
#            samples = data[np.random.random_integers(0,l,nsamples)]
#            print samples.shape
#            lu = self._check_unique(nsamples, self.hashes, samples)
#            print '%i unique values out of %i' % (lu,nsamples) 
#            if lu > best_score:
#                best_score = lu
#                best = self.hashes
#        
#        self.hashes = best

    def train(self,data,stopping_condition):
        cov = np.cov(data.T)
        values,vectors = np.linalg.eig(cov)
        self.hashes = vectors.T[:self.nhashes]
            
    
    def __call__(self,data):
        print data.shape
        print self.hashes.shape
        arr = np.sign(np.dot(self.hashes,data.T).T)
        return arr > 0



class BinaryLsh(Lsh):
    
    
    
    def __init__(self,size,nhashes = 32):
        keys = BinaryLsh.TYPE_CODES.keys()
        if nhashes not in keys:
            raise ValueError('nhashes must be in %s' % str(keys))
        Lsh.__init__(self,size,nhashes = nhashes)
    
    def _check_unique(self,nsamples,hashes,samples):
        results = self(samples)
        return len(np.unique(results))
    
    def __call__(self,data):
        l = data.shape[0]
        # get an l x bits boolean array, representing the coded inputs
        arr = Lsh.__call__(self,data)
        return pack(arr)