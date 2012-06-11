import numpy as np
from util import downsample,downsampled_shape
from analyze.extractor import SingleInput


class Downsample(SingleInput):
    
    def __init__(self,size = None,factor = None ,needs = None, key = None):
        SingleInput.__init__(self,needs = needs,key = key)
        self.size = size
        self.factor = factor
        
    def dim(self,env):
        return np.product(downsampled_shape(self.size,self.factor))
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        data = self.in_data[0].reshape(self.size)
        return downsample(data,self.factor).ravel()
        

class Sum(SingleInput):
    
    def __init__(self,shape = None, dim = None, axis = None, \
                 needs = None, key = None, nframes = 1, step = 1):
        
        SingleInput.__init__(self, needs = needs, key = key, \
                             nframes = nframes, step = step)
        self._dim = dim
        self._axis = axis
        self._shape = shape
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return self.in_data[0].reshape(self._shape).sum(axis = self._axis)

class Max(SingleInput):
    def __init__(self,shape = None, dim = None, axis = None,
                 needs = None, key = None, nframes = 1, step = 1):
        SingleInput.__init__(self,needs = needs, key = key, nframes = nframes, step = step)
        self._dim = dim
        self._shape = shape
        self._axis = axis
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return np.float32

    def _process(self):
        return self.in_data[0].reshape(self._shape).max(axis = self._axis)
        
    