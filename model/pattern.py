from model import Model



    
    
class Pattern(Model):
    '''
    A Pattern is the central concept in zounds.  Patterns can be nested.
    a "leaf" pattern represents a list of ids which point to audio frames.
    A "branch" pattern points to other patterns.
    '''
    
    
    def __init__(self,_id,source,external_id):
        Model.__init__(self)
        
        self.source = None
        self.external_id = None
        self._id = _id

class FilePattern(Pattern):
    '''
    Represents a pattern that has not yet been imported in the form of an audio
    file on disk
    '''
    
    def __init__(self,_id,source,external_id,filename):
        Pattern.__init__(self,_id,source,external_id)
        self.filename = filename

