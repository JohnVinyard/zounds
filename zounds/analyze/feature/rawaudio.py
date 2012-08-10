from __future__ import division
import numpy as np
from zounds.analyze.audiostream import AudioStream
from zounds.analyze.extractor import Extractor
from zounds.nputil import pad,windowed
from zounds.environment import Environment

class AudioSamples(Extractor):
    
    def __init__(self,samplerate,windowsize,stepsize,needs = None):
        Extractor.__init__(self,needs = needs)
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.key = 'audio'
        env = Environment.instance
        self.window = env.window if None is not env.window else \
                     self.oggvorbis(self.windowsize)
        self.iterator = None
    
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
    
    # BUG: This will only work if windows overlap by half. Is it possible to
    # generalize this to be sensitive to the window overlap? 
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
        if not self.iterator:
            self.iterator = self.stream.__iter__()
        out = self.iterator.next() * self.window
        if self.stream.done:
            self.out = None
            self.done = True
        return out
    
    
class AudioFromDisk(AudioSamples):
    
    def __init__(self,samplerate,windowsize,stepsize, filename, needs = None):
        AudioSamples.__init__(self,samplerate,windowsize,stepsize,needs = needs)
        self._init = False
        self.filename = filename
    
    @property
    def stream(self):
        if not self._init:
            self._stream = AudioStream(\
                            self.filename,
                            self.samplerate,
                            self.windowsize,
                            self.stepsize,
                            Environment.instance.chunksize)
            self._init = True
        
        return self._stream
        
    
    
class AudioFromMemory(AudioSamples):
    
    def __init__(self,samplerate,windowsize,stepsize,samples,needs = None):
        AudioSamples.__init__(self,samplerate,windowsize,stepsize,needs = needs)
        self._stream = None
        self.samples = samples
    
    
    class Stream(object):
        
        def __init__(self,samples,chunksize,windowsize,stepsize):
            object.__init__(self)
            self._chunksize = chunksize
            self.windowsize = windowsize
            self.stepsize = stepsize
            self.samples = samples
            self.done = False
        
        @property
        def chunksize(self):
            return self._chunksize
        
        def __iter__(self):
            ls = len(self.samples)
            leftover = np.zeros(0)
            for i in range(0,ls,self.chunksize):
                start = i
                stop = i + self.chunksize
                self.done = stop >= ls
                current = np.concatenate([leftover,self.samples[start:stop]])
                leftover,w = windowed(\
                    current,self.windowsize,self.stepsize,dopad = self.done)
                yield w
        
        
    @property
    def stream(self):
        if not self._stream:
            self._stream = AudioFromMemory.Stream(\
                            self.samples,
                            Environment.instance.chunksize,
                            self.windowsize,self.stepsize)
        
        return self._stream
