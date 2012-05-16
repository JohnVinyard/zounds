import numpy as np
from util import downsample,downsampled_shape
from analyze.extractor import SingleInput


class Downsample(SingleInput):
    
    def __init__(self,size,factor,needs = None, key = None):
        SingleInput.__init__(needs = needs,key = key)
        self.size = size
        self.factor = factor
        
    def dim(self,env):
        return downsampled_shape(self.size,self.factor)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return downsample(self.in_data[0],self.factor)
    