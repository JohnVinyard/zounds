from flow import Node
import numpy as np
from scipy.fftpack import dct
from zounds.analyze.psychoacoustics import Chroma as ChromaScale

class FFT(Node):
    
    def __init__(self, needs = None, axis = -1):
        super(FFT,self).__init__(needs = needs)
        self._axis = axis
    
    def _process(self, data):
        transformed = np.fft.rfft(data, axis = self._axis)
        sl = [slice(None) for _ in xrange(len(transformed.shape))]
        sl[self._axis] = slice(1, None)
        yield np.abs(transformed[sl])

class DCT(Node):
    
    def __init__(self, needs = None, axis = -1):
        super(DCT,self).__init__(needs = needs)
        self._axis = axis
    
    def _process(self, data):
        yield dct(data, norm = 'ortho', axis = self._axis)

class Chroma(Node):
    
    def __init__(\
         self, 
         needs = None, 
         samplerate = 44100., 
         nbins = 12, 
         a440 = 440.):
        
        super(Chroma,self).__init__(needs = needs)
        self._nbins = nbins
        self._a440 = a440
        self._samplerate = samplerate
        self._chroma_scale = None
    
    def _process(self, data):
        if self._chroma_scale is None:
            self._chroma_scale = ChromaScale(\
                 self._samplerate, data.shape[1] * 2, nbands = self._nbins)
        yield self._chroma_scale.transform(data)