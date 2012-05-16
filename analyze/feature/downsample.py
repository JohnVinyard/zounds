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
        
    