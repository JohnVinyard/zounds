from __future__ import division

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis
from scipy.signal import triang
from scipy.fftpack import dct
from scipy.spatial.distance import cdist

from zounds.environment import Environment
from zounds.analyze.extractor import SingleInput
import zounds.analyze.bark as bark
from zounds.util import flatten2d
from zounds.nputil import safe_log,safe_unit_norm as sun


class FFT(SingleInput):
    
    def __init__(self, needs = None, key = None, inshape = None, 
                 nframes = 1, step = 1, axis = -1):
        
        SingleInput.__init__(self, needs = needs, nframes = nframes, 
                             step = step, key = key)
        self._axis = axis if axis < 0 else axis + 1
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
        print data.shape
        data = data.reshape((data.shape[0],) + self._inshape)
        print data.shape
        # return the magnitudes of a real-valued fft along the axis specified,
        # excluding the zero-frequency term
        out = np.abs(np.fft.rfft(data,axis = self._axis)[self._slice])
        print out.shape
        out = flatten2d(out)
        print out.shape
        return out
    
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
        ls = len(self.in_data.shape)
        r = range(ls)
        summed = np.apply_over_axes(np.sum, self.in_data, r[1:])
        if summed.size > 1:
            return summed.squeeze()
        
        return summed.reshape((1,))

class SpectralCentroid(SingleInput):
    '''
    "Indicates where the "center of mass" of the spectrum is. Perceptually, 
    it has a robust connection with the impression of "brightness" of a 
    sound.  It is calculated as the weighted mean of the frequencies 
    present in the signal, determined using a Fourier transform, with 
    their magnitudes as the weights..."
    
    From http://en.wikipedia.org/wiki/Spectral_centroid
    '''
    
    def __init__(self,needs = None,key = None,step = 1):
        SingleInput.__init__(self,needs = needs,key = key, step = step)
        self._bins = None
        self._bins_sum = None
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data
        if self._bins is None:
            self._bins = np.arange(1,spectrum.shape[-1] + 1)
            self._bins_sum = np.sum(self._bins)
        
        return (spectrum*self._bins).sum(axis = 1) / self._bins_sum


class SpectralFlatness(SingleInput):
    '''
    "Spectral flatness or tonality coefficient, also known as Wiener 
    entropy, is a measure used in digital signal processing to characterize an
    audio spectrum. Spectral flatness is typically measured in decibels, and 
    provides a way to quantify how tone-like a sound is, as opposed to being 
    noise-like. The meaning of tonal in this context is in the sense of the 
    amount of peaks or resonant structure in a power spectrum, as opposed to 
    flat spectrum of a white noise. A high spectral flatness indicates that 
    the spectrum has a similar amount of power in all spectral bands - this 
    would sound similar to white noise, and the graph of the spectrum would 
    appear relatively flat and smooth. A low spectral flatness indicates that
    the spectral power is concentrated in a relatively small number of 
    bands - this would typically sound like a mixture of sine waves, and the
    spectrum would appear "spiky"..."
    
    From http://en.wikipedia.org/wiki/Spectral_flatness
    '''
    
    def __init__(self, needs = None, key = None, step = 1):
        SingleInput.__init__(self,needs = needs, key = key, step = step)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data
        m = spectrum.mean(axis = 1)
        return (gmean(spectrum,axis = 1) / m) if m else 0

class Kurtosis(SingleInput):
    
    def __init__(self, needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return kurtosis(self.in_data,axis = 1)
    
class BFCC(SingleInput):
    
    def __init__(self, needs = None, key = None, ncoeffs = 13,exclude = 1):
        SingleInput.__init__(self, needs = needs, key = key)
        self.ncoeffs = ncoeffs
        # the first coefficient is often dropped, as it carries information
        # about the loudness of the signal
        self.exclude = exclude
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.ncoeffs
    
    def _process(self):
        barks = self.in_data
        
        return dct(safe_log(barks),axis = 1)[:,self.exclude: self.exclude + self.ncoeffs]

class AutoCorrelation(SingleInput):
    '''
    Compute the autocorrelation, using the Wiener-Khinchin theorem, detailed
    here: http://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    '''
    def __init__(self, needs = None, key = None, inshape = None):
        SingleInput.__init__(self, needs = needs, key = key)
        if not inshape:
            raise ValueError('please specifiy a size')
        self._inshape = inshape
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._inshape
    
    def _process(self):
        data = self.in_data
        f = np.fft.fft(data,axis=-1)
        f2 = f*f.conjugate()
        return np.fft.ifft(f2,axis = -1)


        
class Difference(SingleInput):
    def __init__(self, needs = None, key = None, size = None):
        SingleInput.__init__(self, needs = needs, key = key)
        if not size:
            raise ValueError('please specifiy a size')
        self.size = size
        self._memory = np.zeros(self.size)
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.size
    
    def _process(self):
        indata = self.in_data
        output =  indata - self._memory
        self._memory = indata
        return output

class Flux(SingleInput):
    
    def __init__(self, needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return ()
    
    def _process(self):
        return np.sqrt((self.in_data**2).sum(axis = 1))
        