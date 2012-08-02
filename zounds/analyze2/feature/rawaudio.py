from __future__ import division
import numpy as np

from zounds.analyze2.audiostream import AudioStream
from zounds.analyze2.extractor import Extractor
from zounds.nputil import pad
from zounds.environment import Environment

class AudioSamples(Extractor):
    
    def __init__(self,samplerate,windowsize,stepsize,needs = None):
        Extractor.__init__(self,needs = needs)
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.key = 'audio'
        env = Environment.instance
        self.window = env.window if None is not env.window else self.oggvorbis(self.windowsize)
        
    
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
            print data
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
