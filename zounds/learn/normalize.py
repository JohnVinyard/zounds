from __future__ import division
import numpy as np

from learn import Learn
from zounds.environment import Environment

class Normalize(Learn):
    '''
    Normalize features based on averages over a representative sample.  In other 
    words, try to remove bias on a per-feature basis.
    '''
    
    def __init__(self,feature):
        Learn.__init__(self)
        self._feature = feature
        c = Environment.instance.framecontroller
        self._vec = np.zeros(c.get_dim(self._feature))
    
    def train(self,data,stopping_condition):
        l = len(Environment.instance.framecontroller)
        for d in data:
            self._vec += d / l
        self._vec = 1 / self._vec
    
    def __call__(self,data):
        return data * self._vec
        
        