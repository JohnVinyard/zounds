from flow import Node
import numpy as np
from zounds.nputil import windowed

# TODO: Steal AudioStream tests and use them here
class SlidingWindow(Node):
    
    def __init__(self, windowsize = None, stepsize = None, needs = None):
        super(SlidingWindow,self).__init__(needs = needs)
        self._windowsize = windowsize
        self._stepsize = stepsize
        self._cache = None
    
    def _enqueue(self,data,pusher):
        if self._cache is None:
            self._cache = data
        else:
            self._cache = np.concatenate([self._cache, data])

    def _dequeue(self):
        leftover, arr = windowed(\
             self._cache,
             self._windowsize,
             self._stepsize, 
             dopad = self._finalized)
        self._cache = leftover
        return arr