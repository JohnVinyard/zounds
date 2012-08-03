from __future__ import division
import numpy as np
from zounds.analyze2.extractor import Extractor,SingleInput
from zounds.analyze2 import chunksize

class MetaDataExtractor(Extractor):
    
    def __init__(self,pattern,key = None,needs = None):
        Extractor.__init__(self,needs = needs,key=key)
        self.pattern = pattern
        self.store = False
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        raise NotImplementedError()
    
    def _process(self):
        l = len(self.input[self.sources[0]])
        out = np.array([self.pattern.data()] * l)
        return out 


class LiteralExtractor(SingleInput):
    
    def __init__(self,dtype,needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
        self._dtype = dtype
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return self._dtype
    
    def _process(self):
        # Possible KLUDGE: Since this extractor has frames = 1 and stepsize = 1,
        # numpy's broadcasting rules will take care of the assignment of this
        # value to any size collection.
        data = self.in_data
        return data[0][self.key]

class CounterExtractor(Extractor):
    
    def __init__(self,needs = None, key = None):
        Extractor.__init__(self,needs = needs, key = key)
        self.n = 0
    
    def dim(self,env):
        return ()

    @property
    def dtype(self):
        return np.int32
    
    def _process(self):
        if self.input[self.sources[0]].shape == ():
            l = 1
        else:
            l = len(self.input[self.sources[0]])
        c = np.arange(self.n,self.n + l)
        self.n += l
        return c
