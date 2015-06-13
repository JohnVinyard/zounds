from flow import Node,NotEnoughData
import numpy as np
from zounds.nputil import windowed

def oggvorbis(s):
    '''
    This is taken from the ogg vorbis spec 
    (http://xiph.org/vorbis/doc/Vorbis_I_spec.html)

    s is the total length of the window, in samples
    '''
    s = np.arange(s)    
    i = np.sin((s + .5) / len(s) * np.pi) ** 2
    f = np.sin(.5 * np.pi * i)
    
    return f * (1. / f.max())

# TODO: Steal AudioStream tests and use them here
class SlidingWindow(Node):
    
    def __init__(\
         self, 
         windowsize = None, 
         stepsize = None, 
         window_func = oggvorbis, 
         needs = None):
        
        super(SlidingWindow,self).__init__(needs = needs)
        self._windowsize = windowsize
        self._stepsize = stepsize
        self._cache = None
        self._window_func = window_func
    
    def _enqueue(self,data,pusher):
        if self._cache is None:
            self._cache = data
        else:
            self._cache = np.concatenate([self._cache, data])

    def _dequeue(self):
        if self._cache.shape[0] < self._windowsize and not self._finalized:
            raise NotEnoughData()
        
        
        leftover, arr = windowed(\
             self._cache,
             self._windowsize,
             self._stepsize, 
             dopad = self._finalized)
        
        
        self._cache = leftover
        
        if not arr.size:
            raise NotEnoughData()
        
        # BUG: Don't recompute the window function every time!
        return \
        (arr * self._window_func(arr.shape[1])) if self._window_func else arr