from __future__ import division
import numpy as np
from analyze.extractor import SingleInput
from nputil import safe_log,safe_unit_norm as sun
from scipy.signal import convolve

class Basic(SingleInput):
    
    def __init__(self, inshape = None, outshape = None, op = None, needs = None, \
                  key = None, nframes = 1, step = 1):
        
        SingleInput.__init__(\
                self,needs = needs, key = key, nframes = nframes, step = step)
        
        self._inshape = inshape
        self._outshape = outshape
        self._op = op
        
    def dim(self,env):
        return self._outshape
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        data = np.array(self.in_data[:self.nframes])
        data = data.reshape(self._inshape)
        return [self._op(data)]



class Abs(Basic):
    
    def __init__(self, inshape = None, needs = None, key = None, 
                 nframes = 1, step = 1):
        
        Basic.__init__(self,inshape = inshape, outshape = np.product(inshape),
                       op = np.abs, needs = needs, key = key, 
                       nframes = nframes, step = step)

class UnitNorm(Basic):
    
    def __init__(self, inshape = None, needs = None, key = None, 
                 nframes = 1, step = 1):
        
        Basic.__init__(self, inshape = inshape, outshape = np.product(inshape),
                       op = sun, needs = needs, key = key,
                       nframes = nframes, step = step)
class Log(Basic):
    
    def __init__(self, inshape = None, needs = None, key = None,
                 nframes = 1, step = 1):
        
        Basic.__init__(self, inshape = inshape, outshape = np.prod(inshape),
                       op = safe_log, needs = needs, key = key,
                       nframes = nframes, step = step)


class SliceX(SingleInput):
    
    def __init__(self, needs = None, key = None, slce = None, 
                 nframes = 1, step = 1):
        
        SingleInput.__init__(self, needs = needs, key = key, 
                             nframes = nframes, step = step)
        if 1 == len(slce):
            self._slice = slice(0,slce[0],1)
        elif 2 == len(slce):
            self._slice = slice(slce[0],slce[1],1)
        else:
            self._slice = slice(*slce)
    
    def dim(self,env):
        return int((self._slice.stop - self._slice.start) / self._slice.step)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return [self.in_data[0][self._slice]]


class Threshold(SingleInput):
    
    def __init__(self,needs = None, key = None, thresh = None, nframes = 1, step = 1):
        SingleInput.__init__(self,needs = needs, key = key, nframes = nframes, step = step)
        self._thresh = thresh
        try:
            self._dim = len(thresh)
        except AttributeError:
            self._dim = ()
    
    @property
    def dtype(self):
        return np.uint8
    
    def dim(self,env):
        return self._dim
    
    def _process(self):
        data = np.array(self.in_data).ravel()
        return [(data > self._thresh).astype(self.dtype)]

class Sharpen(SingleInput):
    '''
    Convolve input with a simple high-pass filter to accentuate peaks
    '''
    
    def __init__(self,needs = None, key = None, nframes = 1, step = 1, 
                 kernel_size = 11,sharpness = 10,inshape = None):
        SingleInput.__init__(self,needs = needs, nframes = nframes,
                             step = step, key = key)
        
        if not kernel_size % 2:
            raise ValueError('kernel_size must be odd')
        
        self._filter = -np.ones(kernel_size)
        self._filter[int(kernel_size / 2)] = sharpness
        self._inshape = inshape
    
    def dim(self,env):
        return self._inshape
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        d = np.array(self.in_data[:self.nframes]).reshape(self._inshape)
        return convolve(d,self._filter,mode = 'same')
        