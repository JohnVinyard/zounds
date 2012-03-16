from __future__ import division
from audiostream import AudioStream
from extractor import Extractor,SingleInput
import numpy as np

# TODO: Implement Pitch, BFCC, Centroid, Flatness, Bark, Tempo, Chroma, 
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

class RawAudio(Extractor):
    
    def __init__(self,samplerate,windowsize,stepsize,needs = None):
        Extractor.__init__(self,needs = needs)
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.key = 'audio'
        self.window = self.oggvorbis(self.windowsize)
        
        self._init = False
    
    def dim(self,env):
        return (self.windowsize,)
    
    @property
    def dtype(self):
        return np.float64
        
    def _process(self):
        
        if not self._init:
            data = self.input[self.sources[0]][0]
            filename = data['filename']
            self.stream = AudioStream(\
                            filename,
                            self.samplerate,
                            self.windowsize,
                            self.stepsize).__iter__()
            self._init = True
        
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

class FFT(SingleInput):
    
    def __init__(self,needs=None,key=None):
        SingleInput.__init__(self,needs=needs,nframes=1,step=1,key=key)
    
    def dim(self,env):
        return int(env.windowsize / 2)
    
    @property
    def dtype(self):
        return np.float64
        
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
        return np.float64
        
    def _process(self):
        return np.sum(self.in_data)
    

    

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
    
    