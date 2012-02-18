from model import Model



    
    
class Pattern(Model):
    '''
    A Pattern is the central concept in zounds.  Patterns can be nested.
    a "leaf" pattern represents a list of ids which point to audio frames.
    A "branch" pattern points to other patterns.
    '''
    
    
    def __init__(self,_id):
        Model.__init__(self)
        
        self.source = None
        
        self.external_id = None
        
        
        # TODO: Move this into Model
        # a unique identifier for this pattern
        self._id = _id
        

