
class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1048
    
class Environment(object):
    '''
    A Zounds client application
    '''
    
    _test = False 
    instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.instance or cls._test:       
            cls.instance = super(Environment, cls).__new__(
                                            cls, *args, **kwargs)
        
        return cls.instance
        
        
 
    
    def __init__(self,
                 source,
                 framemodel,
                 data,
                 audio = AudioConfig):
        
        object.__init__(self)
        
        # audio settings, samplerate, windowsize and stepsize
        self.audio = audio
        
        # the name of the 'source' attribute of patterns. Usually the client
        # application's name
        self.source = source
        
        # a frame.model.Frames derived class that defines the features that
        # the client app considers important
        self.framemodel = framemodel
        
        # a dictionary-like object mapping classes to data backends
        self.data = data
            
    @property
    def windowsize(self):
        return self.audio.windowsize
    
    @property
    def stepsize(self):
        return self.audio.stepsize
    
    @property
    def samplerate(self):
        return self.audio.samplerate
    
    def extractor_chain(self,filename):
        return self.framemodel.extractor_chain(filename=filename)
    
    