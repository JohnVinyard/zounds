from __future__ import division

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis
from scipy.signal import triang
from scipy.fftpack import dct
from scipy.spatial.distance import cdist

from zounds.environment import Environment
from zounds.analyze2.extractor import SingleInput
import zounds.analyze.bark as bark
from zounds.nputil import safe_log,safe_unit_norm as sun


class FFT(SingleInput):
    
    def __init__(self, needs = None, key = None, inshape = None, 
                 nframes = 1, step = 1, axis = -1):
        
        SingleInput.__init__(self, needs = needs, nframes = nframes, 
                             step = step, key = key)
        self._axis = axis
        self._dim = None
        self._slice = None
        if None is inshape or isinstance(inshape,tuple):
            self._inshape = inshape
        elif isinstance(inshape,int):
            self._inshape = (inshape,)
        else:
            raise ValueError('inshape must be None, an int, or a tuple')
        
        if None is not self._inshape:
            a = np.array(self._inshape,dtype = np.float32)
            a[0] = int(a[0] / 2)
            self._dim = int(np.product(a))
        else:
            ws = Environment.instance.windowsize
            self._dim = int(ws / 2)
            self._inshape = (ws,)
        
        # create a list of slices to remove the zero-frequency term along
        # whichever axis we're computing the FFT
        self._slice = [slice(None) for i in xrange(len(self._inshape) + 1)]
        # this slice will remove the zero-frequency term
        self._slice[self._axis] = slice(1,None)

    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._dim
    
    def _process(self):
        data = self.in_data
        # return the magnitudes of a real-valued fft along the axis specified,
        # excluding the zero-frequency term
        return np.abs(np.fft.rfft(data,axis = self._axis)[self._slice])
    
class BarkBands(SingleInput):
    
    
    def __init__(self,needs=None, key=None,
                 nbands = None,start_freq_hz = 50, stop_freq_hz = 20000):
        
        SingleInput.__init__(self,needs=needs, nframes=1, step=1, key=key)
        if None is nbands:
            raise ValueError('an integer must be supplied for nbands')
        
        self.nbands = nbands
        self.env = Environment.instance
        self.samplerate = self.env.samplerate
        self.windowsize = self.env.windowsize
        
        self.start_freq_hz = start_freq_hz
        self.stop_freq_hz = stop_freq_hz
        self.start_bark = bark.hz_to_barks(self.start_freq_hz)
        self.stop_bark = bark.hz_to_barks(self.stop_freq_hz)
        self.bark_bandwidth = (self.stop_bark - self.start_bark) / self.nbands
        self._build_data()

    def dim(self,env):
        return self.nbands
    
    @property
    def dtype(self):
        return np.float32
    
    def _build_data(self):
        # slices of fft coefficients
        self._slices = []
        # triangle windows to multiply the fft slices by
        self._triwins = []
        for i in xrange(1,self.nbands + 1):
            b = i * self.bark_bandwidth
            hz = bark.barks_to_hz(b)
            _herb = bark.erb(hz) / 2.
            start_hz = hz - _herb
            start_hz = 0 if start_hz < 0 else start_hz
            stop_hz = hz + _herb
            s_index,e_index = bark.fft_span(\
                            start_hz,stop_hz,self.windowsize,self.samplerate)
            triang_size = e_index - s_index
            triwin = triang(triang_size)
            self._slices.append(slice(s_index,e_index))
            self._triwins.append(triwin)
    
    def _process(self):
        fft = self.in_data
        return bark.bark_bands(self.samplerate, 
                               self.windowsize, 
                               self.nbands, 
                               self.start_freq_hz, 
                               self.stop_freq_hz, 
                               fft, 
                               fft.shape[0],
                               self._slices,
                               self._triwins)


class Loudness(SingleInput):
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        SingleInput.__init__(self,needs=needs,nframes=nframes,step=step,key=key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
        
    def _process(self):
        return np.sum(self.in_data,axis = -1)

# TODO: Get rid of this
class Mean(SingleInput):
    
    def __init__(self,needs = None, nframes = 1, step = 1, key = None):
        SingleInput.__init__(self,needs = needs, nframes = nframes, step = step, key = key)
    
    def dim(self,env):
        return ()

    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return np.mean(self.in_data,axis = -1)