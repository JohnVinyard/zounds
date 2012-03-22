from __future__ import division

import numpy as np

from scipy.stats.mstats import gmean

from audiostream import AudioStream
from extractor import Extractor,SingleInput
from util import pad

# TODO: Implement Pitch, BFCC, Bark, Tempo, Chroma, 
# Onset, Autocorrelation, DCT

class MetaDataExtractor(Extractor):
    
    def __init__(self,pattern,key = None):
        Extractor.__init__(self,needs = None,key=key)
        self.pattern = pattern
        self.store = False
        self.infinite = True
    
    def dim(self,env):
        raise NotImplementedError()
    
    @property
    def dtype(self):
        raise NotImplementedError()
    
    def _process(self):
        return self.pattern.data()


class LiteralExtractor(SingleInput):
    
    def __init__(self,dtype,needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
        self._dtype = dtype
        self.infinite = True
    
    def dim(self,env):
        return 1
    
    @property
    def dtype(self):
        return self._dtype
    
    def _process(self):
        return self.in_data[0][self.key]

class CounterExtractor(Extractor):
    
    def __init__(self,needs = None, key = None):
        Extractor.__init__(self,needs = needs, key = key)
        self.n = 0
        self.infinite = True
    
    def dim(self,env):
        return ()

    @property
    def dtype(self):
        return np.int32
    
    def _process(self):
        n = self.n
        self.n += 1
        return n

class AudioSamples(Extractor):
    
    def __init__(self,samplerate,windowsize,stepsize,needs = None):
        Extractor.__init__(self,needs = needs)
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.key = 'audio'
        self.window = self.oggvorbis(self.windowsize)
        
        
    
    def dim(self,env):
        return (self.windowsize,)
    
    @property
    def dtype(self):
        return np.float32
    
    @property
    def stream(self):
        raise NotImplemented()
    
    def __hash__(self):
        return hash(\
                    (self.__class__.__name__,
                     self.windowsize,
                     self.stepsize))
    
    def oggvorbis(self,s):
        '''
        This is taken from the ogg vorbis spec 
        (http://xiph.org/vorbis/doc/Vorbis_I_spec.html)
    
        s is the total length of the window, in samples
        '''
        s = np.arange(s)    
        i = np.sin((s + .5) / len(s) * np.pi) ** 2
        f = np.sin(.5 * np.pi * i)
        
        return f * (1. / f.max())
    
    def _process(self):
        
        try:
            return self.stream.next() * self.window
        except StopIteration:
            self.out = None
            self.done = True
            
            # KLUDGE: This is a bit odd.   The RawAudio extractor is telling
            # an extractor on which it depends that it is done.  This is 
            # necessary because the MetaData extractor generates data with
            # no source. It has no idea when to stop.
            self.sources[0].out = None
            self.sources[0].done = True
    
    
class AudioFromDisk(AudioSamples):
    
    def __init__(self,samplerate,windowsize,stepsize, needs = None):
        AudioSamples.__init__(self,samplerate,windowsize,stepsize,needs = needs)
        self._init = False
    
    @property
    def stream(self):
        if not self._init:
            data = self.input[self.sources[0]][0]
            # KLUDGE: I shouldn't have to know a specific name here
            filename = data['filename']
            self._stream = AudioStream(\
                            filename,
                            self.samplerate,
                            self.windowsize,
                            self.stepsize).__iter__()
            self._init = True
        
        return self._stream
        
    
    
class AudioFromMemory(AudioSamples):
    
    def __init__(self,samplerate,windowsize,stepsize,needs = None):
        AudioSamples.__init__(self,samplerate,windowsize,stepsize,needs = needs)
        self._init = False
        
    def get_stream(self):
        '''
        A generator that returns a sliding window along samples
        '''
        for i in xrange(0,len(self.samples),self.stepsize):
            yield pad(self.samples[i : i + self.windowsize],self.windowsize)
    
    @property
    def stream(self):
        if not self._init:
            data = self.input[self.sources[0]][0]
            # KLUDGE: I shouldn't have to know a specific name here
            self.samples = data['samples']
            self._stream = self.get_stream()
            self._init = True
        
        return self._stream

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
    

from extractor import ExtractorChain
from matplotlib import pyplot as plt
import optparse


if __name__ == '__main__':
    parser = optparse.OptionParser()
    aa = parser.add_option
    aa('--filename', help='path to a wav, aiff, or flac file', dest='filename')
    options, args = parser.parse_args()
    
    raw = RawAudio(options.filename)
    fft = FFT(needs=raw)
    loud = Loudness(needs=fft,nframes=2,step=2)
    d = ExtractorChain([loud,raw,fft]).collect()
    
    fftdata = np.array(d[fft])
    fftdata = fftdata[:,:200]
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(np.rot90(np.log(fftdata)))
    plt.subplot(2,1,2)
    plt.plot(d[loud])
    plt.show()
    plt.savefig('features.png')
    plt.clf()
    
    