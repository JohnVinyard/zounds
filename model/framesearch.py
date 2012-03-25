from abc import ABCMeta,abstractmethod

from model import Model
from pattern import DataPattern

class FrameSearch(Model):
    '''
    FrameSearch-derived classes use the frames backing store to provide results
    to queries in the form of sound using one or more stored features.
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self,*features):
        Model.__init__(self)
        
    
    @abstractmethod
    def search(self,audio):
        p = DataPattern(None,None,None,audio)
        ec = self.env().framemodel.extractor_chain(p)
        
        
    