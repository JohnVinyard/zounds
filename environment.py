from uuid import uuid4
from celery.task import subtask,chord
from analyze.synthesize import WindowedAudioSynthesizer

class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1024
    window = None


class Environment(object):
    '''
    A Zounds client application
    '''
    
    _test = False 
    instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.instance or cls._test:       
            cls.instance = super(Environment, cls).__new__(cls)
        
        return cls.instance
        
        
 
    
    def __init__(self,
                 source,
                 framemodel,
                 framecontroller,
                 framecontroller_args,
                 data,
                 audio = AudioConfig):
        
        object.__init__(self)
        
        # audio settings, samplerate, windowsize and stepsize
        self.audio = audio
        
        # A synthesizer that can create raw audio samples from the encoding
        # stored in the 'audio' feature of frames.  Note that for now, the
        # "encoding" and "decoding" are really no-ops, except for the windowing
        # function on the encoding side
        self.synth = WindowedAudioSynthesizer(self.windowsize,self.stepsize)
        
        # Should we do analysis and db syncing in multiple processes?
        self.parallel = False
        
        # the name of the 'source' attribute of patterns. Usually the client
        # application's name
        self.source = source
        
        # a frame.model.Frames derived class that defines the features that
        # the client app considers important
        self.framemodel = framemodel
        
        self.framecontroller_class = framecontroller
        
        # a dictionary-like object mapping classes to data backends
        self.data = data
        
        self.framecontroller = framecontroller(*framecontroller_args)
        self.data[framemodel] = self.framecontroller
        if not Environment._test:
            self.framemodel.sync()
    
    def play(self,audio):
        self.synth.play(audio)
    
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
    
    def newid(self):
        return uuid4().hex
    
    def extractor_chain(self,pattern):
        return self.framemodel.extractor_chain(pattern)
    
    def append(self,pattern):
        ec = self.extractor_chain(pattern)
        self.framecontroller.append(ec)
    
    def __repr__(self):
        return '''Environment(
    source     : %s,
    samplerate : %i,
    windowsize : %i,
    stepsize   : %i,
    data       : %s 
)
''' % (self.source,self.samplerate,self.windowsize,self.stepsize,self.data)

    def __str__(self):
        return self.__repr__()
    
    