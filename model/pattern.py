from model import Model


class MetaPattern(Model):
        
    @property
    def controller(self):
        return config.data[self]
    
    
class Pattern(object):
    '''
    A pattern is the central concept in zounds.  Patterns can be nested.
    a "leaf" pattern represents a list of ids which point to audio frames.
    A "branch" pattern points to other patterns.
    '''
    __metaclass__ = MetaPattern
    
    class FrameSequence(list):
        '''
        Represents frames in time. List members
        can be integer ids or slices.
        '''
        def __init__(self):
            list.__init__(self)
            
        def compress(self):
            '''
            Compress the sequence into as many
            slices and as few individual elements
            as possible
            '''
            pass
        
        def fetch(self,datastore):
            '''
            Call compress, and then fetch
            '''
            pass
        
    
    def __init__(self,_id):
        object.__init__(self)
        
        # a unique identifier for this pattern
        self._id = _id
        
        # a list of frame ids and/or two-tuples that can synthesize
        # this pattern.
        self.frames = Pattern.FrameSequence()
        
import config
