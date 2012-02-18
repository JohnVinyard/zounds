
class Environment(object):
    '''
    A Zounds client application
    '''
    
    instance = None
    def __new__(cls, *args, **kwargs):
        
        if not cls.instance:
            cls.instance = super(Environment, cls).__new__(
                                cls, *args, **kwargs)
            return cls.instance
        
        return cls.instance
    
    def __init__(self,
                 source,
                 framemodel,
                 data,
                 audio):
        
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
            