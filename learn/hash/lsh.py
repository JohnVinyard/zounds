import struct

import numpy as np
from scipy.cluster.vq import whiten,kmeans
from scipy.spatial.distance import cdist
from bitarray import bitarray

from learn.learn import Learn

class Lsh(Learn):
    
    def __init__(self,size,nhashes = 32):
        Learn.__init__(self)
        self.size = size
        self.nhashes = nhashes
        self.hashes = None
    
    def _check_unique(self,nsamples,hashes,samples):
        self.hashes = hashes.reshape((len(hashes),np.product(hashes.shape[1:])))
        samples = samples.reshape((len(samples),np.product(samples.shape[1:])))
        results = np.zeros((nsamples,self.nhashes))
        for i in range(nsamples):
            results[i] = self(samples[i])
        
        s = set()
        [s.add(tuple(r)) for r in results]
        return len(s)
        
    
    def train(self,data,stopping_condition):
        '''
        Assume centered data that is in a completely randomized order.
        Look for hash functions (example data) that approximately splits
        randomly selected data in half.
        '''
        nsamples = 10000
        best_score = 0
        best = None
        l = len(data) - 1
        for i in range(100):
            self.hashes = data[np.random.random_integers(0,l,self.nhashes)]
            print self.hashes.shape
            samples = data[np.random.random_integers(0,l,nsamples)]
            print samples.shape
            lu = self._check_unique(nsamples, self.hashes, samples)
            print '%i unique values out of %i' % (lu,nsamples) 
            if lu > best_score:
                best_score = lu
                best = self.hashes
        
        self.hashes = best  
            
    
    def __call__(self,data):
        arr = np.sign(np.dot(self.hashes,data))
        return arr > 0



class BinaryLsh(Lsh):
    
    TYPE_CODES = {
                  # 32-bit unisgned integer
                  32 : 'L',
                  # 64-bit unsigned integer
                  64 : 'Q'
                  }
    
    def __init__(self,size,nhashes = 32):
        keys = BinaryLsh.TYPE_CODES.keys()
        if nhashes not in keys:
            raise ValueError('nhashes must be in %s' % str(keys))
        Lsh.__init__(self,size,nhashes = nhashes)
        self._type_code = BinaryLsh.TYPE_CODES[nhashes]
    
    def _check_unique(self,nsamples,hashes,samples):
        results = np.zeros(nsamples)
        for i in range(nsamples):
            results[i] = self(samples[i])
        return len(np.unique(results))
    
    def __call__(self,data):
        arr = Lsh.__call__(self,data)
        b = bitarray()
        b.extend(arr)
        return struct.unpack(self._type_code,b.tobytes())[0]
    