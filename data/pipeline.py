from abc import ABCMeta,abstractmethod
import os.path
import cPickle
 
from controller import Controller

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
        pass
    
    @abstractmethod
    def store(self,pipeline):
        pass
    


class PickledLearningController(LearningController):
    '''
    A learning controller that persists Pipelines by pickling them to disk
    '''
    
    extension = '.dat'
    
    def __init__(self,directory):
        Controller.__init__()
    
    def _filename(self,_id):
        return '%s%s' % (_id,PickledLearningController.extension)
    
    def __getitem__(self,key):
        filename = self._filename(key)
        with open(filename,'rb') as f:
            return cPickle.load(f)
    
    def __delitem__(self,key):
        raise NotImplemented()
    
    def store(self,pipeline):
        # TODO: Ensure path exists method in util. Factor out of 
        # PyTablesFrameController and this
        filename = self._filename(pipeline._id)
        parts = os.path.split(filename)
        path = os.path.join(*parts[:-1])
        
        try: 
            os.makedirs(path)
        except OSError:
            # This probably means that the path already exists
            pass
        
        with open(filename,'wb') as f:
            cPickle.dump(pipeline,f)
            
        
        
        