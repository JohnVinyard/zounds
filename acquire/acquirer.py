from abc import ABCMeta,abstractmethod,abstractproperty
from environment import Environment
from celery.tasks import task

import os

class Acquirer(object):
    '''
    Acquirers fetch sound files from disk, or remote locations (e.g., 
    Freesound.org) and enqueue them for processing and insertion into the
    database
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractproperty
    def source(self):
        '''
        The source name, e.g. 'Freesound' or 'StemsFromMyAlbum'
        '''
        pass
    
    @abstractmethod
    def acquire(self):
        '''
        Do the work necessary to acquire sound data, process it, and
        insert it into the db
        '''
        pass
    
class DiskAcqurer(Acquirer):
    
    def __init__(self,path,source = None):
        Acquirer.__init__(self)
        self.path = path
        self._source = source
        # TODO: If source is none, make source the
        # final section of the path
    
    @property
    def source(self):
        return self._source
    
    def acquire(self):
        for fn in os.listdir(self.path):
            fp = os.path.join(self.path,fn)
            
    
    