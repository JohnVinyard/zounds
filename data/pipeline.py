from abc import ABCMeta,abstractmethod
import os.path
import cPickle
 
from controller import Controller

class PipelineController(Controller):
    '''
    An abstract base class for controllers that will persist and fetch
    learning pipelines
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self):
        Controller.__init__(self)
    
    @abstractmethod
    def __delitem__(self):
        pass
    
    @abstractmethod    
    def __getitem__(self,key):
        pass
    
    @abstractmethod
    def store(self,pipeline):
        pass

class DictPipelineController(PipelineController):
    '''
    A learning controller that only persists Pipelines in memory
    '''
    
    def __init__(self):
        PipelineController.__init__(self)
        self._store = {}
    
    def __getitem__(self,key):
        return cPickle.loads(self._store[key])

    def __delitem__(self,key):
        del self._store[key]
    
    def store(self,pipeline):
        self._store[pipeline._id] = cPickle.dumps(pipeline,cPickle.HIGHEST_PROTOCOL)


class PickledPipelineController(PipelineController):
    '''
    A learning controller that persists Pipelines by pickling them to disk
    '''
    
    extension = '.dat'
    
    def __init__(self):
        Controller.__init__(self)
    
    def _filename(self,_id):
        return '%s%s' % (_id,PickledPipelineController.extension)
    
    def __getitem__(self,key):
        filename = self._filename(key)
        try:
            with open(filename,'rb') as f:
                return cPickle.load(f)
        except IOError:
            raise KeyError
    
    def __delitem__(self,key):
        path = self._filename(key)
        os.remove(path)
            
    
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
            cPickle.dump(pipeline,f,cPickle.HIGHEST_PROTOCOL)
            
        
        
        