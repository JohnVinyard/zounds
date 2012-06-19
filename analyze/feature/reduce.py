import numpy as np
from util import downsample,downsampled_shape
from analyze.extractor import SingleInput
from basic import Basic

class Downsample(SingleInput):
    
    def __init__(self,size = None,factor = None ,needs = None, 
                 key = None, step = 1, nframes = 1):
        
        SingleInput.__init__(self,needs = needs,key = key,
                             step = step, nframes = nframes)
        if 2 != len(size):
            raise ValueError('Downsample expects a 2d input')
        
        self.size = size
        self.factor = factor
        
    def dim(self,env):
        return np.product(downsampled_shape(self.size,self.factor))
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        data = np.array(self.in_data[:self.nframes]).reshape(self.size)
        return downsample(data,self.factor).ravel()
        



class Reduce(Basic):
    
    def __init__(self,inshape = None, op = None, axis = None, needs = None, 
                 key = None, nframes = 1, step = 1):
        
        _inshape = [nframes] if nframes > 1 else []
        try:
            _inshape.extend(inshape)
        except TypeError:
            _inshape.append(inshape)
        
        _op = lambda a : op(a,axis = axis)
        
        sh = list(_inshape)
        sh.pop(axis)
        _dim = np.product(sh)
        _dim = () if 1 == _dim else _dim
        Basic.__init__(self,inshape = _inshape, outshape = _dim, op = _op, 
                       needs = needs, key = key, nframes = nframes, 
                       step = step)
    

    

class Sum(Reduce):
    
    def __init__(self, inshape = None, axis = 0, needs = None, 
                 key = None, nframes = 1, step = 1):
        
        Reduce.__init__(self,inshape = inshape, op = np.sum, axis = axis, 
                        needs = needs, key = key, nframes = nframes, 
                        step = step)

class Max(Reduce):
    
    def __init__(self, inshape = None, axis = None, needs = None, 
                 key = None, nframes = 1, step = 1):
        
        Reduce.__init__(self,inshape = inshape, axis = axis, op = np.max,
                        needs = needs, key = key, nframes = nframes, 
                        step = step)
        
        

