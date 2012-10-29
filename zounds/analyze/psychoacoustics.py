from __future__ import division
from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.signal import triang

def fft_index(freq_hz,ws,sr,rnd = np.round):
    '''
    Given a frequency in hz, a window size, and a sample rate,
    return the fft bin into which the freq in hz falls
    '''
    if freq_hz < 0 or freq_hz > sr / 2.: 
        raise ValueError(\
            'Freq must be greater than zero and less than the Nyquist frequency')
    
    fft_bandwidth = (sr * .5) / (ws * .5)
    return int(rnd(freq_hz / fft_bandwidth))

def fft_span(start_hz,stop_hz,ws,sr):
    '''
    Given a span in hz, return the start and stop fft bin indices covering
    that span
    '''
    s_index = fft_index(start_hz,ws,sr, rnd = np.floor)
    e_index = fft_index(stop_hz,ws,sr, rnd = np.ceil)
    # ensure that the span is at least one fft bin
    e_index = s_index + 1 if s_index == e_index else e_index
    return s_index,e_index

def _hz_is_valid(freq_hz,sr):
    if freq_hz < 0 or freq_hz > sr / 2.: 
        raise ValueError(\
            'Freq must be greater than zero and less than the Nyquist frequency')

# TODO: Should this be different for Bark and Mel scales?  Isn't equivalent
# rectangular bandwidth based on the Bark scale?
def erb(hz):
    '''
    equivalent rectangular bandwidth
    '''
    return (0.108 * hz) + 24.7

class Scale(object):
    
    __metaclass__ = ABCMeta
    
    _data = dict()
    
    def __init__(self,samplerate,window_size,nbands,start_freq_hz,stop_freq_hz):
        object.__init__(self)
        self._fft_bandwidth = (samplerate * .5) / (window_size * .5)
        self._sr = samplerate
        self._ws = window_size
        self._nb = nbands
        self._sfhz = start_freq_hz
        self._efhz = stop_freq_hz
        self._data_key = (self.__class__.__name__,
                          self._sr,self._ws,self._nb,
                          self._sfhz,self._efhz)
        
        if self._data_key not in self._data:
            self._data[self._data_key] = self._build_data() 
        
    @abstractmethod
    def to_hz(self,s):
        pass
    
    @abstractmethod
    def from_hz(self,hz):
        pass
    
    def _build_data(self):
        start_unit = self.from_hz(self._sfhz)
        stop_unit = self.from_hz(self._efhz)
        bandwidth = (stop_unit - start_unit) / self._nb
        
        _slices = []
        _triwins = []
        
        for i in xrange(1,self._nb + 1):
            b = i * bandwidth
            hz = self.to_hz(b)
            _herb = erb(hz) / 2
            start_hz = hz - _herb
            start_hz = 0 if start_hz < 0 else start_hz
            stop_hz = hz + _herb
            _hz_is_valid(start_hz,self._sr)
            _hz_is_valid(stop_hz,self._sr)
            s = start_hz / self._fft_bandwidth
            e = stop_hz / self._fft_bandwidth
            s_index = int(np.floor(s))
            e_index = int(np.ceil(e))
            e_index = s_index + 1 if s_index == e_index else e_index
            n_bins = e_index - s_index
            _slices.append(slice(s_index,e_index))
            
            tri = triang(n_bins)
            peak_diff = (n_bins // 2) / (n_bins / 2) 
            bin_diff = (e - s) / n_bins
            
            # scale the triangular window to account for aliasing introduced by
            # a) the rounded number of bins in the window
            # b) the fact that even numbered windows won't have a peak with value
            #    1 in the middle
            tri = tri * peak_diff * bin_diff
            _triwins.append(tri)
        
        return _slices,_triwins
    
    @property
    def n_bands(self):
        return self._nb
    
    @property
    def data(self):
        return self._data[self._data_key]
    
    def transform(self,fft):
        slices,triwins = self.data
        cb = np.ndarray((fft.shape[0],self._nb),dtype = np.float32)
        for i in xrange(self._nb):
            cb[:,i] = (fft[:,slices[i]] * triwins[i]).sum(1)
        return cb
    
class Bark(Scale):
    
    def __init__(self,samplerate,window_size,nbands,start_freq_hz,stop_freq_hz):
        Scale.__init__(\
                self,samplerate,window_size,nbands,start_freq_hz,stop_freq_hz)
    
    def to_hz(self,b):
        return 300. * ((np.e ** (b/6.0)) - (np.e ** (-b/6.)))
    
    def from_hz(self,hz):
        return 6. * np.log((hz/600.) + np.sqrt((hz/600.)**2 + 1))

class Mel(Scale):
    
    def __init__(self,samplerate,window_size,nbands,start_freq_hz,stop_freq_hz):
        Scale.__init__(\
                self,samplerate,window_size,nbands,start_freq_hz,stop_freq_hz)
    
    def to_hz(self,m):
        return 700 * ((np.e**(m/1127)) - 1)
    
    def from_hz(self,hz):
        return 1127 * np.log(1 + (hz/700))
    
    