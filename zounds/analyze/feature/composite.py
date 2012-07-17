import numpy as np
from analyze.extractor import Extractor


class Composite(Extractor):
    '''
    Combine the output of multiple extractors into a single feature
    '''
    def __init__(self,step = 1, nframes = 1,needs = None, key = None):
        Extractor.__init__(\
                self,step = step, nframes = nframes, needs = needs, key = key)
    
    
    def dim(self,env):
        dims = [int(np.product(s.dim(env)) * self.nframes) for s in self.sources]
        return np.sum(dims)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return np.concatenate([np.array(self.input[source]).ravel() \
                               for source in self.sources]).ravel()
    
    