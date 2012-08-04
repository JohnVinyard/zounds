from abc import ABCMeta,abstractmethod,abstractproperty
import os.path
from zounds.util import audio_files

from zounds.environment import Environment
from zounds.model.pattern import FilePattern



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
    def _acquire(self):
        pass
    
    def acquire(self):
        '''
        Do the work necessary to acquire sound data, process it, and
        insert it into the db
        '''
        # let subclasses do the dirty work
        self._acquire()
        # update indexes, if applicable for this data store
        self.framecontroller.update_index()
    
class DiskAcquirer(Acquirer):
    
    def __init__(self,path,source = None):
        Acquirer.__init__(self)
        self.path = os.path.normpath(path)
        self._source = source if source else os.path.split(self.path)[1]
    
    @property
    def source(self):
        return self._source
    
    def _acquire(self):
        files = audio_files(self.path)
        lf = len(files)
        for i,fn in enumerate(files):
            fp = os.path.join(self.path,fn)
            extid = os.path.splitext(fn)[0]
            pattern = FilePattern(self.env.newid(),
                                  self.source,
                                  extid,
                                  fp)
            if not self.framecontroller.exists(self.source,extid):
                try:
                    print 'importing %s, %s, file %i of %i' % \
                             (self.source,fn,i,lf)
                    self.framecontroller.append(self.extractor_chain(pattern))
                except IOError:
                    print 'ERROR! : data from %s was unreadable' % fn
            else:
                print 'skipping %s' % fn
            
            
            
    
    