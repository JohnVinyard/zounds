from controller import Controller
from abc import ABCMeta,abstractmethod
from tables import openFile,IsDescription,StringCol,Int32Col,Col
import os.path
import re
import numpy as np
import time
from util import pad

class FrameController(Controller):
    __metaclass__ = ABCMeta
    
    def __init__(self,framesmodel):
        Controller.__init__(self)
        self.model = framesmodel
    
     
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
    
    @abstractmethod
    def get_dtype(self,key):
        '''
        Get the data type of the feature with key
        '''
        pass
    
    @abstractmethod
    def get_dim(self,key):
        '''
        Get the dimension of the feature with key
        '''
        pass


# TODO: Write documentation
class PyTablesFrameController(FrameController):
    
    # TODO: How do I switch between read and write modes as necessary in a
    # transparent manner?
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self,framesmodel)
        self.filepath = filepath
        parts = os.path.split(filepath)
        self.filename = parts[-1]
        path = os.path.join(*parts[:-1])
        
        self.dbfile_write = None
        self.db_write = None
        
        self.dbfile_read = None
        self.db_read = None
        
        if len(parts) > 1:
            try:
                os.makedirs(path)
            except OSError:
                # This probably means that the path already exists
                pass
            
        # KLUDGE: PyTables allows the creation of columns using a string
        # representation of a datatype, e.g. "float32", but not the numpy
        # type representing that type, e.g., np.float32.  This is a hackish
        # way of extracting a string representation that PyTables can
        # understand from a numpy type
        rgx = re.compile('(\'|^)(numpy\.)?(?P<type>[a-z0-9]+)(\'|$)')
        # TODO: Write tests for this method!
        def get_type(np_dtype):
            m = rgx.search(str(np_dtype))
            if not m:
                raise ValueError('Unknown dtype %s' % str(np_dtype))
            return m.groupdict()['type']
        
        
        # create the table's schema from the FrameModel
        self.steps = {}
        desc = {}
        pos = 0
        dim = self.model.dimensions()    
        for k,v in dim.iteritems():
            t = get_type(v[1])
            if t.startswith('a') or t.startswith('|S'):
                # This is a string feature
                desc[k] = Col.from_kind('string',
                                        itemsize=np.dtype(t).itemsize,
                                        pos=pos)
            else:
                # This is a numeric feature
                desc[k] = Col.from_type(t,shape=v[0],pos=pos)
            self.steps[k] = v[2]
            pos += 1
        
        if not os.path.exists(filepath):
            
            self.dbfile_write = openFile(filepath,'w')
            
            # create the table
            self.dbfile_write.createTable(self.dbfile_write.root, 'frames', desc)
            
            self.db_write = self.dbfile_write.root.frames
            
            # create indices for any string column or one-dimensional
            # numeric column
            for k,v in desc.iteritems():
                col = getattr(self.db_write.cols,k)
                oned = 1 == len(col.shape)
                if isinstance(col,StringCol) or oned:
                    col.createIndex()
                    
            self.dbfile_write.close()
            

        self.dbfile_read = openFile(filepath,'r')
        self.db_read = self.dbfile_read.root.frames
        
        # TODO: Consider moving this out of __init__
        # create our buffer
        def lcd(numbers):
            i = 1
            while any([i % n for n in numbers]):
                i += 1
            return i
        
        self._desired_buffer_size = 5000
        # once we've processed this much data, stop and wait to write it
        self._max_buffer_size = self._desired_buffer_size * 5
        
        # find the lowest common multiple of all step sizes
        l = lcd(self.steps.values())
        # find a whole number multiple of the lowest common
        # multiple that puts us close to our desired buffer size
        self._buffer_size = l * int(self._desired_buffer_size / l)
        self.recarray_dtype = []
        for k in self.db_read.colnames:
            col = getattr(self.db_read.cols,k)
            self.recarray_dtype.append((k,col.dtype,col.shape[1:]))
         
        self.has_lock = False
    
    def to_recarray(self,d,rootkey):
        '''
        Convert a dictionary of extracted features into a numpy.recarray 
        suitable to be passed to PyTables.Table.append()
        '''
        # the rootkey must have a stepsize of one
        l = len(d[rootkey])
        buf = np.recarray(l,dtype=self.recarray_dtype)
        for k,v in d.iteritems():
            try:
                data = pad(np.array(v).repeat(self.steps[k], axis = 0),l)
                buf[k] = data
            except KeyError:
                # This feature isn't stored, so it isn't in the steps
                # dictionary
                pass
        return buf
        
        
    def append(self,chain,rootkey):
        '''
        Turn the crank on an extractor chain until it runs out of data. Persist
        data to the hdf5 file in chunks as we go.
        '''
        bufsize = self._buffer_size
        
        bucket = dict([(c.key if c.key else c,[]) for c in chain])
        nframes = 0
        for k,v in chain.process():
            if rootkey == k and \
                (nframes == bufsize or nframes >= self._max_buffer_size):
                
                # we've reached our smallest buffer size. Let's attempt a write
                try:
                    self.acquire_lock(nframes)
                    # we got the lock. Let's write the data we have
                    record = self.to_recarray(bucket, rootkey)
                    self._append(record)
                    bucket = dict([(c.key if c.key else c,[]) for c in chain])
                    nframes = 0
                except PyTablesFrameController.WriteLockException:
                    # someone else has the write lock. Let's just keep processing
                    # for awhile (within reason)
                    bufsize += self._buffer_size
                    
            if rootkey == k:
                nframes += 1
            
            bucket[k].append(v)
            
        # We've processed the entire file. Wait until we can get the write lock    
        self.acquire_lock(nframes,wait=True)
        # build the record and append it
        record = self.to_recarray(bucket, rootkey)
        self._append(record)
        
        # release the lock for the next guy
        self.release_lock()    
        
    
    @property
    def lock_filename(self):
        return self.filepath + '.lock'
    
    class WriteLockException(BaseException):
        
        def __init__(self):
            BaseException.__init__(self)
        
    def acquire_lock(self,nframes,wait=False):
        
        if self.has_lock:
            return
        
        locked = os.path.exists(self.lock_filename) 
        if locked and\
             (not wait) and \
             (nframes < self._max_buffer_size):
            # Someone else has the lock, but the wait is False, 
            # and we haven't yet reached our max buffer size.
            raise PyTablesFrameController.WriteLockException()
        
        # Either we've reached our maximum buffer size, or we've been
        # explicitly instructed to wait for the lock
        while locked:
            time.sleep(1)
            locked = os.path.exists(self.lock_filename)
        
        f = open(self.lock_filename,'w')
        f.close()
        self.has_lock = True
        
    
    def release_lock(self):
        os.remove(self.lock_filename)
        self.has_lock = False
        
    def _append(self,frames):
        
        # switch to write mode
        self.close()
        self.dbfile_write = openFile(self.filepath,'a')
        self.db_write = self.dbfile_write.root.frames
        
        # append the rows
        self.db_write.append(frames)
        self.db_write.flush()
        
        # switch back to read mode
        self.close()
        self.dbfile_read = openFile(self.filepath,'r')
        self.db_read = self.dbfile_read.root.frames
        
        
        
    
    def close(self):
        if self.dbfile_write:
            self.dbfile_write.close()
        if self.dbfile_read:
            self.dbfile_read.close()
    
    def __del__(self):
        self.close()
        
    
    def check(self,framesmodel):
        raise NotImplemented()
    
    
    def sync(self,add,update,delete,chain):
        raise NotImplemented()
    
    
    def get(self,indices,features=None):
        raise NotImplemented()
  
    
    def get_features(self):
        raise NotImplemented()
    
    
    def set_features(self):
        raise NotImplemented()
    
    
    def get_dtype(self,key):
        raise NotImplemented()
    
    
    def get_dim(self,key):
        raise NotImplemented()
         
        

class DictFrameController(FrameController):
    
    def __init__(self,framesmodel):
        FrameController.__init__(self,framesmodel)
        
    def check(self,framesmodel):
        raise NotImplemented()
    
    
    def sync(self,add,update,delete,chain):
        raise NotImplemented()
    
    
    def append(self,frames):
        raise NotImplemented()
    
    
    def get(self,indices,features=None):
        raise NotImplemented()
  
    
    def get_features(self):
        raise NotImplemented()
    
    
    def set_features(self):
        raise NotImplemented()
    
    
    def get_dtype(self,key):
        raise NotImplemented()
    
    
    def get_dim(self,key):
        raise NotImplemented()
    
    
        
        