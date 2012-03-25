from abc import ABCMeta,abstractmethod

from model import Model
from pattern import DataPattern

# TODO: I'm not sure the model module package is the appropriate place for this,
# and/or if it should be Model-derived
class FrameSearch(Model):
    '''
    FrameSearch-derived classes use the frames backing store to provide results
    to queries in the form of sound using one or more stored features.
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self,*features):
        Model.__init__(self)
        self.features = features
    
    @abstractmethod
    def _search(self,frames):
        '''
        Do work
        '''
        pass
    
    def search(self,audio):
        p = DataPattern(None,None,None,audio)
        fm = self.env().framemodel
        # TODO: ExtractorChain.prune() that can take multiple features and tests
        ec = fm.extractor_chain(p).prune(self.features)
        d = ec.collect()
        # TODO: This isn't possible yet. Frames-derived classes must be
        # instantiatable? from data that isn't stored
        frames = fm(d)
        return self._search(frames)
    
     
        
        
        
    