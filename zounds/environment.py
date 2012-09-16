from __future__ import division
from uuid import uuid4
from multiprocessing import cpu_count

from zounds.analyze.synthesize import WindowedAudioSynthesizer
from zounds.util import tostring


class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1024
    window = None


class Environment(object):
    '''
    A Zounds client application
    '''
    
    n_cores = cpu_count()
    _test = False 
    instance = None
    
    @classmethod
    def parallel(cls):
        return not cls._test and cls.n_cores > 1
    
    def __new__(cls, *args, **kwargs):
        if not cls.instance or cls._test:       
            cls.instance = super(Environment, cls).__new__(cls)
        
        return cls.instance
        
    
    def __init__(self,source,framemodel,framecontroller,framecontroller_args,
                data,audio = AudioConfig,chunksize_seconds = 45.,do_sync = True):
        
        
        object.__init__(self)
        
        self.do_sync = do_sync
        
        # audio settings, samplerate, windowsize and stepsize
        self.audio = audio
        
        self.chunksize_seconds = chunksize_seconds
        # processing chunk size, in samples
        self.chunksize = chunksize_seconds * self.samplerate
        # approximate number of absolute frames in each chunk
        self.chunksize_frames = int(self.chunksize / self.stepsize)
        
        # A synthesizer that can create raw audio samples from the encoding
        # stored in the 'audio' feature of frames.  Note that for now, the
        # "encoding" and "decoding" are really no-ops, except for the windowing
        # function on the encoding side
        self.synth = WindowedAudioSynthesizer(self.windowsize,self.stepsize)
        
        
        # the name of the 'source' attribute of patterns. Usually the client
        # application's name
        self.source = source
        
        # a frame.model.Frames derived class that defines the features that
        # the client app considers important
        self.framemodel = framemodel
        
        self.framecontroller_class = framecontroller
        self._framecontroller_args = framecontroller_args
        
        self._data = dict([(k,v.__class__) for k,v in data.iteritems()])
        # a dictionary-like object mapping classes to data-backends instances
        self.data = data
        
        self.framecontroller = framecontroller(*framecontroller_args)
        self.data[framemodel] = self.framecontroller
        if not Environment._test and self.do_sync:
            self.framemodel.sync()
    
    def __repr__(self):
        return tostring(self,
                        short = False,
                        sample_rate = self.samplerate,
                        window_size = self.windowsize,
                        step_size = self.stepsize,
                        chunksize_seconds = self.chunksize_seconds,
                        source = self.source,
                        framemodel = self.framemodel,
                        framecontroller = self.framecontroller_class,
                        data = self.data)
    
    def __str__(self):
        return tostring(self,
                        sample_rate = self.samplerate,
                        window_size = self.windowsize,
                        step_size = self.stepsize,
                        source = self.source,
                        framemodel = self.framemodel)
    
    def play(self,audio,block = True):
        self.synth.play(audio,block = block)
    
    def shush(self):
        self.synth.shush()
    
    @property
    def address_class(self):
        return self.framecontroller_class.Address
        
    @property
    def windowsize(self):
        return self.audio.windowsize
    
    @property
    def stepsize(self):
        return self.audio.stepsize
    
    @property
    def samplerate(self):
        return self.audio.samplerate
    
    @property
    def window(self):
        return self.audio.window
    
    def seconds_to_frames(self,secs):
        return int((secs * self.samplerate) / self.stepsize)
    
    def frames_to_seconds(self,nframes):
        if not nframes:
            return 0
        
        overlap = self.windowsize - self.stepsize
        return (nframes * (self.stepsize / self.samplerate)) +\
                 (overlap / self.samplerate)
    
    def newid(self):
        return uuid4().hex
    
    def extractor_chain(self,pattern):
        return self.framemodel.extractor_chain(pattern)
    
    def append(self,pattern):
        ec = self.extractor_chain(pattern)
        self.framecontroller.append(ec)
    
    # KLUDGE: This is specific to PyTables, i.e., there are lots of problems
    # with concurrent reads.  This should be handled some other way.
    def unique_controller(self):
        return self.framecontroller_class(*self._framecontroller_args)
    
    def __getstate__(self,lock = None,**kwargs):
        d = {'source' : self.source,
             'framemodel' : self.framemodel,
             'framecontroller' : self.framecontroller_class,
             'framecontroller_args' : self._framecontroller_args,
             'data' : self._data,
             'audio' : self.audio,
             'chunksize_seconds' : self.chunksize_seconds,
             'do_sync' : self.do_sync}
        
        # KLUDGE: This assumes that lock will always be the final argument
        # to the FrameController
        if None is not lock:
            d['framecontroller_args'] += (lock,)
        
        for k,v in kwargs.iteritems():
            d[k] = v
        
        return d
    
    def __setstate__(self,d):
        data = d['data']
        for k,v in data.iteritems():
            # KLUDGE: This assumes that controllers will always have zero-arg
            # constructors
            #
            # BUG: For some reason, when n_cores = 2, the values in d['data']
            # come back as controller instances instead of classes when the
            # second round of pattern chunks are processed
            data[k] = v() if isinstance(v,type) else v

        return Environment(d['source'],
                           d['framemodel'],
                           d['framecontroller'],
                           d['framecontroller_args'],
                           data,
                           d['audio'],
                           d['chunksize_seconds'],
                           d['do_sync'])
    
    @staticmethod
    def reanimate(data):
        env = Environment.__new__(Environment)
        env = env.__setstate__(data)
        return env
    
    