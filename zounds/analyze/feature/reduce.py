import numpy as np
from zounds.util import downsample,downsampled_shape,flatten2d
from zounds.nputil import sliding_window
from zounds.analyze.extractor import SingleInput
from basic import Basic

class Downsample(SingleInput):
    '''
    Downsample a 2d input
    '''
    
    def __init__(self,inshape = None,factor = None ,needs = None, key = None, 
                 step = 1, nframes = 1):
        
        SingleInput.__init__(self,needs = needs,key = key,
                             step = step, nframes = nframes)
        if 2 != len(inshape):
            raise ValueError('Downsample expects a 2d input')
        
        self._inshape = inshape 
        self._dim = np.product(downsampled_shape(self._inshape,factor)) 
        
    def dim(self,env):
        return self._dim 
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        data = self.in_data
        l = data.shape[0]
        data = data.reshape((l,) + self._inshape)
        shape = (1,) + self._dim
        chunks = sliding_window(data,shape,flatten = False).squeeze()
        # downsample the chunks by taking their average
        downsampled = np.apply_over_axes(np.mean, chunks, (-1,-2)).squeeze()
        return flatten2d(downsampled)
        



class Reduce(Basic):
    
    def __init__(self,inshape = None, op = None, axis = None, needs = None, 
                 key = None, nframes = 1, step = 1):
        
        _inshape = [nframes] if nframes > 1 else []
        try:
            _inshape.extend(inshape)
        except TypeError:
            _inshape.append(inshape)
        
        # all dimensions are given in terms of single examples. In reality,
        # the operation will happen over multiple examples.  If the axis is
        # negative, this should still work ok. If the axis is zero or positive,
        # advance it by one so the reduction happens over the correct axis in 
        # the _process method 
        realaxis = axis + 1 if axis >= 0 else axis 
        _op = lambda a : op(a,axis = realaxis)
        
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
    
    def __init__(self, inshape = None, axis = 0, needs = None, 
                 key = None, nframes = 1, step = 1):
        
        Reduce.__init__(self,inshape = inshape, axis = axis, op = np.max,
                        needs = needs, key = key, nframes = nframes, 
                        step = step)

class Min(Reduce):
    
    def __init__(self, inshape = None, axis = 0, needs = None, 
                 key = None, nframes = 1, step = 1):
        
        Reduce.__init__(self,inshape = inshape, axis = axis, op = np.min,
                        needs = needs, key = key, nframes = nframes, 
                        step = step)
        
        

