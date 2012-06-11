import numpy as np
from analyze.extractor import SingleInput


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
        data = np.reshape(self.in_data[:self.nframes],self._inshape)
        return [self._op(data)]



class Abs(Basic):
    
    def __init__(self, inshape = None, needs = None, key = None, nframes = 1, step = 1):
        
        Basic.__init__(self,inshape = inshape, outshape = np.product(inshape),
                       op = np.abs, needs = needs, key = key, 
                       nframes = nframes, step = step)
        