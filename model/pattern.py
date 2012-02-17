from model import Model


class MetaPattern(Model):
        
    @property
    def controller(self):
        return config.data[self]
    
    
class Pattern(object):
    '''
    A Pattern is the central concept in zounds.  Patterns can be nested.
    a "leaf" pattern represents a list of ids which point to audio frames.
    A "branch" pattern points to other patterns.
    '''
    __metaclass__ = MetaPattern
    
    
    def __init__(self,_id):
        object.__init__(self)
        
        self.source = None
        
        self.external_id = None
        
        
        # TODO: Move this into Model
        # a unique identifier for this pattern
        self._id = _id
        

        
import config
