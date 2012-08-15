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
    
    def train(self,data,stopping_condition):
        cov = np.cov(data.T)
        values,vectors = np.linalg.eig(cov)
        self.hashes = vectors.T[:self.nhashes]
            
    
    def __call__(self,data):
        arr = np.sign(np.dot(self.hashes,data.T).T)
        return arr > 0



class BinaryLsh(Lsh):
    
    
    
    def __init__(self,size,nhashes = 32):
        keys = BinaryLsh.TYPE_CODES.keys()
        if nhashes not in keys:
            raise ValueError('nhashes must be in %s' % str(keys))
        Lsh.__init__(self,size,nhashes = nhashes)
    
    
    def __call__(self,data):
        l = data.shape[0]
        # get an l x bits boolean array, representing the coded inputs
        arr = Lsh.__call__(self,data)
        return pack(arr)