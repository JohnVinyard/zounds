from flow import Node
import numpy as np
from scipy.fftpack import dct
from zounds.analyze.psychoacoustics import \
    Chroma as ChromaScale, Bark as BarkScale
from zounds.nputil import safe_log
from zounds.node.timeseries import ConstantRateTimeSeries
from zounds.node.samplerate import SR44100

class FFT(Node):
    
    def __init__(self, needs = None, axis = -1):
        super(FFT,self).__init__(needs = needs)
        self._axis = axis
    
    def _process(self, data):
        transformed = np.fft.fft(data, axis = self._axis)
        sl = [slice(None) for _ in xrange(len(transformed.shape))]
        positive = data.shape[self._axis] // 2
        sl[self._axis] = slice(positive, None)
        yield ConstantRateTimeSeries(\
             transformed[sl], 
             data.frequency, 
             data.duration)

class DCT(Node):
    
    def __init__(self, needs = None, axis = -1):
        super(DCT,self).__init__(needs = needs)
        self._axis = axis
    
    def _process(self, data):
        yield ConstantRateTimeSeries(\
             dct(data, norm = 'ortho', axis = self._axis),
             data.frequency,
             data.duration)

# TODO: This constructor should not take a samplerate; that information should
# be encapsulated in the data that's passed in
class Chroma(Node):
    
    def __init__(\
         self, 
         needs = None, 
         samplerate = SR44100(), 
         nbins = 12, 
         a440 = 440.):
        
        super(Chroma,self).__init__(needs = needs)
        self._nbins = nbins
        self._a440 = a440
        self._samplerate = samplerate
        self._chroma_scale = None
    
    def _process(self, data):
        data = np.abs(data)
        if self._chroma_scale is None:
            self._chroma_scale = ChromaScale(\
                 self._samplerate.samples_per_second, 
                 data.shape[1] * 2, 
                 nbands = self._nbins)
        
        yield ConstantRateTimeSeries(\
             self._chroma_scale.transform(data),
             data.frequency,
             data.duration)

# TODO: This constructor should not take a samplerate; that information should
# be encapsulated in the data that's passed in
class BarkBands(Node):
    
    def __init__(\
         self, 
         needs = None,
         samplerate = SR44100(), 
         n_bands = 100, 
         start_freq_hz = 50, 
         stop_freq_hz = 2e4):
        super(BarkBands,self).__init__(needs = needs)
        
        self._samplerate = samplerate
        self._n_bands = n_bands
        self._start_freq_hz = start_freq_hz
        self._stop_freq_hz = stop_freq_hz
        self._bark_scale = None
    
    def _process(self, data):
        data = np.abs(data)
        if self._bark_scale is None:
            self._bark_scale = BarkScale(\
                 self._samplerate.samples_per_second, 
                 data.shape[1] * 2, 
                 self._n_bands,
                 self._start_freq_hz,
                 self._stop_freq_hz)
        yield ConstantRateTimeSeries(\
             self._bark_scale.transform(data),
             data.frequency,
             data.duration)

class BFCC(Node):
    
    def __init__(self, needs = None, n_coeffs = 13, exclude = 1):
        super(BFCC,self).__init__(needs = needs)
        self._n_coeffs = n_coeffs
        self._exclude = exclude
    
    def _process(self, data):
        data = np.abs(data)
        bfcc = dct(safe_log(data),axis = 1)\
            [:,self._exclude : self._exclude + self._n_coeffs]
        yield ConstantRateTimeSeries(\
             bfcc.copy(), data.frequency, data.duration)
    
        