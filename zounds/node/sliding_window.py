from flow import Node, NotEnoughData
import numpy as np
from zounds.nputil import windowed
from timeseries import Picoseconds

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

class OggVorbisWindowingFunc(WindowingFunc):
    
    def __init__(self):
        super(OggVorbisWindowingFunc, self).__init__()
    
    def _wdata(self, size):
        return oggvorbis(size)

class WindowingScheme(object):
    
    def __init__(self, duration, frequency):
        self.duration = duration
        self.frequency = frequency

class HalfLapped(WindowingScheme):
    
    def __init__(self):
        one_sample_at_44100 = Picoseconds(int(1e12)) / 44100.
        window = one_sample_at_44100 * 2048
        step = window / 2
        super(HalfLapped, self).__init__(window, step)

class SlidingWindow(Node):
    
    def __init__(self, wscheme, wfunc, needs = None):
        super(SlidingWindow, self).__init__(needs = needs)
        self._scheme = wscheme
        self._func = wfunc
        self._cache = None
    
    def _enqueue(self, data, pusher):
        if self._cache is None:
            self._cache = data
            self._windowsize = int(self._scheme.duration / data.frequency)
            self._stepsize = int(self._scheme.frequency / data.frequency)
        else:
            self._cache = np.concatenate([self._cache, data])
    
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
        return (self._func * arr) if self._func else arr