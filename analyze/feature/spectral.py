from __future__ import division

import numpy as np

from scipy.stats.mstats import gmean
from scipy.signal import triang
from scipy.fftpack import dct

from environment import Environment
from analyze.extractor import SingleInput
from analyze import bark



class FFT(SingleInput):
    
    def __init__(self,needs=None,key=None):
        SingleInput.__init__(self,needs=needs,nframes=1,step=1,key=key)
    
    def dim(self,env):
        return int(env.windowsize / 2)
    
    @property
    def dtype(self):
        return np.float32
        
    def _process(self):
        '''
        Return the magnitudes only, discarding phase and the zero
        frequency component
        '''
        return np.abs(np.fft.rfft(self.in_data[0]))[1:]

class BarkBands(SingleInput):
    
    _triang = {}
    _fft_index = {}
    _hz_to_barks = {}
    _barks_to_hz = {}
    _erb = {}
    
    def __init__(self,nbands = None,needs=None, key=None):
        SingleInput.__init__(self,needs=needs, nframes=1, step=1, key=key)
        if None is nbands:
            raise ValueError('an integer must be supplied for nbands')
        
        self.nbands = nbands
        self.env = Environment.instance
        self.samplerate = self.env.samplerate
        self.windowsize = self.env.windowsize
        
        self.start_freq_hz = 50
        self.stop_freq_hz = 20000
        self.start_bark = bark.hz_to_barks(self.start_freq_hz)
        self.stop_bark = bark.hz_to_barks(self.stop_freq_hz)
        self.bark_bandwidth = (self.stop_bark - self.start_bark) / self.nbands

    def dim(self,env):
        return self.nbands
    
    @property
    def dtype(self):
        return np.float32
    
    @classmethod
    def from_cache(cls,cache,callable,key):
        try:
            return cache[key]
        except KeyError:
            v = callable(key)
            cache[key] = v
            return v
        
    @classmethod
    def fft_index(cls,t):
        return bark.fft_index(*t)
    
    def _process(self):
        cb = np.ndarray(self.nbands,dtype=np.float32)
        for i in xrange(1,self.nbands + 1):
            b = i * self.bark_bandwidth
            hz = BarkBands.from_cache(BarkBands._barks_to_hz, 
                                      bark.barks_to_hz, 
                                      b)
            _erb = BarkBands.from_cache(BarkBands._erb, bark.erb, hz)
            _herb = _erb / 2.
            start_hz = hz - _herb
            stop_hz = hz + _herb
            ws = self.windowsize
            sr = self.samplerate
            s_index = BarkBands.from_cache(BarkBands._fft_index,
                                           BarkBands.fft_index,
                                           (start_hz,ws,sr))
            e_index = BarkBands.from_cache(BarkBands._fft_index,
                                           BarkBands.fft_index,
                                           (stop_hz,ws,sr))
            triang_size = e_index - s_index
            triwin = self.from_cache(BarkBands._triang,triang,triang_size)
            fft_frame = self.in_data[0]
            cb[i - 1] = \
                (fft_frame[s_index : e_index] * triwin).sum()
        
        return cb

class Loudness(SingleInput):
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        SingleInput.__init__(self,needs=needs,nframes=nframes,step=step,key=key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
        
    def _process(self):
        return np.sum(self.in_data)

# TODO : Factor out spectral mean
class SpectralCentroid(SingleInput):
    '''
    "Indicates where the "center of mass" of the spectrum is. Perceptually, 
    it has a robust connection with the impression of "brightness" of a 
    sound.  It is calculated as the weighted mean of the frequencies 
    present in the signal, determined using a Fourier transform, with 
    their magnitudes as the weights..."
    
    From http://en.wikipedia.org/wiki/Spectral_centroid
    '''
    
    def __init__(self,needs = None,key = None):
        SingleInput.__init__(self,needs = needs,key = key)
        self._bins = None
        self._bins_sum = None
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data[0]
        if self._bins is None:
            self._bins = np.arange(1,len(spectrum) + 1)
            self._bins_sum = np.sum(self._bins)
            
        return np.sum(spectrum*self._bins) / self._bins_sum

# TODO : Factor out spectral mean
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
    
    def __init__(self, needs = None, key = None):
        SingleInput.__init__(self,needs = needs, key = key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data[0]
        return gmean(spectrum) / np.average(spectrum)

class BFCC(SingleInput):
    
    def __init__(self, needs = None, key = None, ncoeffs = 13):
        SingleInput.__init__(self, needs = needs, key = key)
        self.ncoeffs = ncoeffs
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.ncoeffs
    
    def _process(self):
        barks = self.in_data[0]
        barks[barks == 0] = .00000001
        return dct(np.log(barks))[:self.ncoeffs]
        
    
    