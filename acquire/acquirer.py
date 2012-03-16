from abc import ABCMeta,abstractmethod,abstractproperty
import os.path

from environment import Environment
from model.pattern import FilePattern



class Acquirer(object):
    '''
    Acquirers fetch sound files from disk, or remote locations (e.g., 
    Freesound.org) and enqueue them for processing and insertion into the
    database
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
        self.env = Environment.instance
    
    @property
    def framemodel(self):
        return self.env.framemodel
    
    @property
    def framecontroller(self):
        return self.framemodel.controller()
    
    def extractor_chain(self,pattern):
        return self.framemodel.extractor_chain(pattern)
    
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
    
class DiskAcquirer(Acquirer):
    
    def __init__(self,path,source = None):
        Acquirer.__init__(self)
        self.path = path
        self._source = source if source else os.path.split(self.path)[1]
        
    
    @property
    def source(self):
        return self._source
    
    def acquire(self):
        for fn in os.listdir(self.path):
            fp = os.path.join(self.path,fn)
            pattern = FilePattern(self.env.newid(),
                                  self.source,
                                  os.path.splitext(fn)[0],
                                  fp)
            self.framemodel.append(self.extractor_chain(pattern))
            
            
    
    