from __future__ import division
import numpy as np
from analyze.extractor import SingleInput
from nputil import safe_unit_norm as sun


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
    
    def __init__(self, inshape = None, needs = None, key = None, nframes = 1, step = 1):
        
        Basic.__init__(self,inshape = inshape, outshape = np.product(inshape),
                       op = np.abs, needs = needs, key = key, 
                       nframes = nframes, step = step)

class UnitNorm(Basic):
    
    def __init__(self, inshape = None, needs = None, key = None, nframes = 1, step = 1):
        
        Basic.__init__(self, inshape = inshape, outshape = np.product(inshape),
                       op = sun, needs = needs, key = key,
                       nframes = nframes, step = step)

class SliceX(SingleInput):
    
    def __init__(self, needs = None, key = None, slce = None):
        SingleInput.__init__(self, needs = needs, key = key, nframes = 1, step = 1)
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