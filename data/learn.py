from controller import Controller
from abc import ABCMeta,abstractmethod

class LearningController(Controller):
    '''
    An abstract base class for controllers that will persist and fetch
    learning pipelines
    '''
    
    __metaclass__ = ABCMeta
    
    
    def __init__(self):
        Controller.__init__(self)
    
    @abstractmethod    
    def __getitem__(self,key):
        raise NotImplemented()
    
    @abstractmethod
    def __setitem__(self,key,value):
        raise NotImplemented()