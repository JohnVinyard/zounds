import cPickle
import os
from zounds.util import ensure_path_exists



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
        '''
        Fetch the item with key. If it doesn't exist, raise a KeyError.
        
        :param key: a hashable object, usually a string
        '''
        filename = self._filename(key)
        try:
            with open(filename,'rb') as f:
                return cPickle.load(f)
        except IOError:
            raise KeyError(key)
    
    def __delitem__(self,key):
        '''
        Delete the item (from disk). If it doesn't exist, raise a KeyError.
        
        :param key: a hashable object, usually a string
        '''
        path = self._filename(key)
        try:
            os.remove(path)
        except OSError:
            raise KeyError(key)
    
    def store(self,item):
        '''
        Store an item, using its :code:`_id` attribute as the item's key. If an
        item with the same :code:`_id` already exists, raise a ValueEror
        
        :param item: a python object with an :code:`_id` attribute
        '''
        filename = self._filename(item._id)
        if os.path.exists(filename):
            raise ValueError(\
                'An object with the key %s already exists. Please delete it\
                 first if you\'d like to store it again' % item._id)
        
        ensure_path_exists(filename)
        
        with open(filename,'wb') as f:
            cPickle.dump(item,f,cPickle.HIGHEST_PROTOCOL)
            