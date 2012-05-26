import numpy as np
from analyze.extractor import Extractor


class Composite(Extractor):
    '''
    Combine the output of multiple extractors into a single feature
    '''
    def __init__(self,dim, step = 1, nframes = 1,needs = None, key = None):
        Extractor.__init__(\
                self,step = step, nframes = nframes, needs = needs, key = key)
        self._dim = dim
    
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return np.array([self.input[source] for source in self.sources]).ravel()
    
    