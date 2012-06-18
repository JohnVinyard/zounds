import cPickle
import os

from util import ensure_path_exists

class Controller(object):
    '''
    Base class for all data controllers. These classes are responsible for
    persisting and retrieving objects defined in model.
    '''
    def __init__(self):
        object.__init__(self)
        
        

class PickledController(Controller):
    '''
    A controller that persists python objects to disk, and provides some
    convenience functions that treat the file system as a key-value store.
    '''
    
    extension = '.dat'
    
    def __init__(self):
        Controller.__init__(self)
    
    def _filename(self,_id):
        return '%s%s' % (_id,self.__class__.extension)
    
    def id_exists(self,_id):
        return os.path.exists(self._filename(_id))
    
    def __getitem__(self,key):
        filename = self._filename(key)
        try:
            with open(filename,'rb') as f:
                return cPickle.load(f)
        except IOError:
            raise KeyError(key)
    
    def __delitem__(self,key):
        path = self._filename(key)
        try:
            os.remove(path)
        except OSError:
            raise KeyError(key)
            
    
    def store(self,pipeline):
        filename = self._filename(pipeline._id)
        if os.path.exists(filename):
            raise ValueError(\
                'An object with the key %s already exists. Please delete it\
                 first if you\'d like to store it again' % pipeline._id)
        
        ensure_path_exists(filename)
        
        with open(filename,'wb') as f:
            cPickle.dump(pipeline,f,cPickle.HIGHEST_PROTOCOL)
            