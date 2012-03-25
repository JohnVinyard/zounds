from abc import ABCMeta,abstractmethod

from model.frame import Feature

class Fetch(object):
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def __call__(self):
        pass

class PrecomputedFeature(Fetch):
    
    def __init__(self,nframes,*features):
        Fetch.__init__(self)
        self.nframes = nframes
        self.features = features
    
    
    
    