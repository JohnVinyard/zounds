from __future__ import division

import numpy as np

from scipy.stats.mstats import gmean
from scipy.signal import triang

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
    
    def _process(self):
        cb = np.ndarray(self.nbands)
        for i in xrange(1,self.nbands + 1):
            b = i * self.bark_bandwidth
            hz = bark.barks_to_hz(b)
            _erb = bark.erb(hz)
            _herb = _erb / 2.
            s_index = \
                bark.fft_index(hz - _herb,self.windowsize,self.samplerate)
            e_index = \
                bark.fft_index(hz + _herb,self.windowsize,self.samplerate) + 1
            fft_frame = self.in_data[0]
            cb[i - 1] = \
                (fft_frame[s_index : e_index] * triang(e_index - s_index)).sum()
        
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
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data[0]
        # TODO: This is wasteful. Get the shape of the source and cache it
        bins = np.arange(1,len(spectrum) + 1)
        return np.sum(spectrum*bins) / np.sum(bins)

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