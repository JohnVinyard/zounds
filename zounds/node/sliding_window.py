from flow import Node, NotEnoughData
import numpy as np
from zounds.nputil import windowed
from timeseries import ConstantRateTimeSeries

def oggvorbis(s):
    '''
    This is taken from the ogg vorbis spec 
    (http://xiph.org/vorbis/doc/Vorbis_I_spec.html)

    s is the total length of the window, in samples
    '''
    try:
        s = np.arange(s)
    except TypeError:
        s = np.arange(s[0])
        
    i = np.sin((s + .5) / len(s) * np.pi) ** 2
    f = np.sin(.5 * np.pi * i)    
    return f * (1. / f.max())

class WindowingFunc(object):    
    
    def _wdata(self, size):
        return np.ones(size)
    
    def __mul__(self, other):
        size = other.shape[1:] 
        return self._wdata(size) * other
    
    def __rmul__(self, other):
        return self.__mul__(other)

class IdentityWindowingFunc(WindowingFunc):
    
    def __init__(self):
        super(IdentityWindowingFunc, self).__init__()

class OggVorbisWindowingFunc(WindowingFunc):
    
    def __init__(self):
        super(OggVorbisWindowingFunc, self).__init__()
    
    def _wdata(self, size):
        return oggvorbis(size)

class SlidingWindow(Node):
    
    def __init__(self, wscheme, wfunc = None, needs = None):
        super(SlidingWindow, self).__init__(needs = needs)
        self._scheme = wscheme
        self._func = wfunc or IdentityWindowingFunc()
        self._cache = None
    
    def _enqueue(self, data, pusher):
        if self._cache is None:
            self._cache = data
            # BUG: I Think this may only work in cases where frequency and
            # duration are the same
            self._windowsize = \
                int((self._scheme.duration - data.overlap) / data.frequency)
            self._stepsize = int(self._scheme.frequency / data.frequency)
        else:
            np.concatenate([self._cache, data])
    
    def _dequeue(self):
        leftover, arr = windowed(\
             self._cache,
             self._windowsize,
             self._stepsize, 
             dopad = self._finalized)

        self._cache = leftover
        
        if not arr.size:
            raise NotEnoughData()
        
        # BUG: Order matters here (try arr * self._func instead)
        # why does that statement result in __rmul__ being called for each
        # scalar value in arr?
        out = (self._func * arr) if self._func else arr
        out = ConstantRateTimeSeries(\
              out, self._scheme.frequency, self._scheme.duration)
        return out