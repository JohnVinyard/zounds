from controller import Controller
from abc import ABCMeta,abstractmethod

class FrameController(Controller):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        Controller.__init__(self)
    
    @abstractmethod
    def in_sync(self,framesmodel):
        '''
        Returns true if the the FrameModel is in sync with the data store,
        i.e., all the features defined on the FrameModel with store = True
        are represented in the db
        '''
        pass
    
    @abstractmethod
    def sync(self,framesmodel):
        '''
        Checks if the database and the FrameModel are in sync (see the in_sync
        method). If they are not, adds any features represented by framesmodel
        but not in the database, using pre-computed data whenever possible.
        '''
        pass
    
    @abstractmethod
    def append(self,frames):
        '''
        Adds frames to the datastore
        '''
        pass
    
    @abstractmethod
    def get(self,indices,features=None):
        '''
        Gets rows, and optionally specific features from those rows.
        Indices may be a single index, a list of indices, or a slice.
        features may be a single feature or a list of them.
        '''
        pass
  

class DictFrameController(FrameController):
    
    def __init__(self):
        FrameController.__init__(self)
    
    def in_sync(self,framesmodel):
        pass  