from abc import ABCMeta,abstractmethod

import cPickle
from controller import Controller,PickledController
from zounds.model.pipeline import Pipeline

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
    
    def id_exists(self,_id):
        return _id in self._store
    
    def store(self,pipeline):
        self._store[pipeline._id] = cPickle.dumps(pipeline,cPickle.HIGHEST_PROTOCOL)


class PickledPipelineController(PickledController,PipelineController):
    '''
    A learning controller that persists Pipelines by pickling them to disk
    '''
        
        
        