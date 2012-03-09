from controller import Controller
from abc import ABCMeta,abstractmethod

class FrameController(Controller):
    __metaclass__ = ABCMeta
    
    def __init__(self,framesmodel):
        Controller.__init__(self)
        self.metadata = framesmodel.features
    
    
    @abstractmethod
    def check(self,framesmodel):
        '''
        Returns true if the the FrameModel is in sync with the data store,
        i.e., all the features defined on the FrameModel with store = True
        are represented in the db
        '''
        pass
    
    @abstractmethod
    def sync(self,add,update,delete,chain):
        '''
        Follows an update plan in order to update the database in the most
        efficient way possible.
        
        This might be a long running process, so we should be able to save state
        and resume in case of an error
        '''
        
        # BUG: This will start to break as the size of the data grows, or
        # if an exception occurs!  I can get around this by creating a task
        # queue:
        # 1) Create a new db
        # 2) Process all sounds
        # 3) Store new representation in new database
        # 4) Delete old database
        # 5) Rename new database
        
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
  
    @abstractmethod
    def get_features(self):
        '''
        Return the current set of features represented in the database
        '''
        pass
    
    @abstractmethod
    def set_features(self):
        '''
        Set the current set of features represented in the database
        '''
        pass

class DictFrameController(FrameController):
    
    def __init__(self,framesmodel):
        FrameController.__init__(self,framesmodel)
    
    
        
        