import struct

import numpy as np
from bitarray import bitarray

from learn.learn import Learn

class Lsh(Learn):
    
    def __init__(self,size,nhashes = 32):
        self.size = size
        self.nhashes = nhashes
        self.mean = None
        self.hashes = None
    
    def train(self,data,stopping_condition):
        self.mean = data.mean(0)
        data -= self.mean
        self.hashes = np.random.permutation(data)[:self.nhashes]
    
    def __call__(self,data):
        data -= self.mean
        arr = np.array([np.dot(h,(data - self.mean)) for h in self.hashes]) > 1
        b = bitarray()
        b.extend(arr)
        # KLUDGE: This has to be sensitive to nhashes!
        return struct.unpack('L',b.tobytes())[0]