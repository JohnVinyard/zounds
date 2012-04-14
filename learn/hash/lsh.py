import struct

import numpy as np
from scipy.cluster.vq import whiten,kmeans
from bitarray import bitarray

from learn.learn import Learn

class Lsh(Learn):
    
    def __init__(self,size,nhashes = 32):
        self.size = size
        self.nhashes = nhashes
        self.hashes = None
    
    def train(self,data,stopping_condition):
        data = whiten(data)
        codebook,distortion = kmeans(data,self.nhashes)
        self.hashes = codebook
    
    def __call__(self,data):
        arr = np.dot(self.hashes,data) > 1
        b = bitarray()
        b.extend(arr)
        # KLUDGE: The datatype must correspond to nhashes
        return struct.unpack('L',b.tobytes())[0]