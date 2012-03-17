import os.path
import re
import time
import cPickle
from abc import ABCMeta,abstractmethod

from tables import openFile,IsDescription,StringCol,Int32Col,Col,Int8Col

import numpy as np

from controller import Controller
from model.pattern import Pattern
from util import pad

class FrameController(Controller):
    __metaclass__ = ABCMeta
    
    def __init__(self,framesmodel):
        Controller.__init__(self)
        self.model = framesmodel
        
    @abstractmethod
    def __len__(self):
        '''
        Return the total number of rows
        '''
        pass
    
    @abstractmethod
    def list_ids(self):
        '''
        List all pattern ids
        '''
        pass
    
    @abstractmethod
    def external_id(self,_id):
        '''
        Return a two-tuple of source,external_id for this _id
        '''
        pass
    
    @abstractmethod
    def exists(self,source,external_id):
        '''
        Return true if a pattern with source and external_id already exists in
        the datastore
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
    def append(self,extractor_chain):
        '''
        Adds frames to the datastore
        '''
        pass
    
    @abstractmethod
    def get(self,_id,features=None):
        '''
        Gets rows, and optionally specific features from those rows.
        Indices may be a single index, a list of indices, or a slice.
        features may be a single feature or a list of them.
        '''
        pass
    
    @abstractmethod
    def iter_feature(self,_id,feature):
        '''
        Return an iterator over a single feature from pattern _id
        '''
        pass
  
    @abstractmethod
    def get_features(self):
        '''
        Return the current set of features represented in the database
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


class PyTablesUpdateNotCompleteError(BaseException):
    '''
    Raised when a PyTables update fails
    '''
    def __init__(self):
        BaseException.__init__(self,Exception('The PyTables update failed'))


# TODO: Cleaner, general way to do PyTables in-kernel queries
class PyTablesFrameController(FrameController):
    
    '''
    A FrameController that stores feature data in the hdf5 file format, and 
    uses the PyTables library to access it.
    
    PyTables has some special limitations, .e.g, columns cannot be added or
    removed after table creation. This class attempts to hide some of the messy
    details from clients
    '''
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self,framesmodel)
        self._load(filepath)
    
    def _load(self,filepath):
        '''
        
        '''
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
            
            # create the table
            self.dbfile_write = openFile(filepath,'w')
            self.dbfile_write.createTable(self.dbfile_write.root, 'frames', desc)
            self.db_write = self.dbfile_write.root.frames
            
            # create a table to store our schema as a pickled byte array
            class FrameSchema(IsDescription):
                bytes = Int8Col(pos = 0)
            
            self.dbfile_write.createTable(\
                        self.dbfile_write.root,'schema',FrameSchema)
            self.schema_write = self.dbfile_write.root.schema
            s = cPickle.dumps(\
                        self.model.stored_features(),cPickle.HIGHEST_PROTOCOL)
            binary = np.fromstring(s,dtype = np.int8)
            record = np.recarray(len(binary),dtype=[('bytes',np.int8)])
            record['bytes'] = binary
            self.schema_write.append(record)
            
            
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
        self.schema_read = self.dbfile_read.root.schema
        
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
    
    def to_recarray(self,d,chain):
        '''
        Convert a dictionary of extracted features into a numpy.recarray 
        suitable to be passed to PyTables.Table.append()
        '''
        # the first extractor that isn't infinite
        e = filter(lambda e : not e.infinite,chain)[0]
        # the length of the first finite extractor. We're assuming that it
        # has a stepsize of 1
        l = len(d[e.key])
        buf = np.recarray(l,dtype=self.recarray_dtype)
        for k,v in d.iteritems():
            try:
                data = np.array(v).repeat(self.steps[k], axis = 0)
                data = pad(data,l) if len(data) <= l else data[:l]
                buf[k] = data
            except KeyError:
                # This feature isn't stored, so it isn't in the steps
                # dictionary
                pass
        return buf
        
        
    def append(self,chain):
        '''
        Turn the crank on an extractor chain until it runs out of data. Persist
        data to the hdf5 file in chunks as we go.
        '''
        bufsize = self._buffer_size
        # the root extractor's key
        rootkey = chain[0].key
        bucket = dict([(c.key if c.key else c,[]) for c in chain])
        nframes = 0
        for k,v in chain.process():
            if rootkey == k and \
                (nframes == bufsize or nframes >= self._max_buffer_size):
                
                # we've reached our smallest buffer size. Let's attempt a write
                try:
                    self.acquire_lock(nframes)
                    # we got the lock. Let's write the data we have
                    record = self.to_recarray(bucket, chain)
                    self._append(record)
                    bucket = dict([(c.key if c.key else c,[]) for c in chain])
                    nframes = 0
                except PyTablesFrameController.WriteLockException:
                    # someone else has the write lock. Let's just keep 
                    # processing for awhile (within reason)
                    bufsize += self._buffer_size
                    
            if rootkey == k:
                nframes += 1
            
            bucket[k].append(v)
            
        # We've processed the entire file. Wait until we can get the write lock    
        self.acquire_lock(nframes,wait=True)
        # build the record and append it
        record = self.to_recarray(bucket, chain)
        print 'appending %i rows' % len(record)
        self._append(record)
        
        # release the lock for the next writer
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
            # Someone else has the lock, but we haven't been explicity
            # instructed to wait, and we haven't yet reached our max buffer 
            # size.
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
    
    def _write_mode(self):
        '''
        switch to write mode
        '''
        self.close()
        self.dbfile_write = openFile(self.filepath,'a')
        self.db_write = self.dbfile_write.root.frames
    
    def _read_mode(self):
        '''
        switch back to read mode
        '''
        self.close()
        self.dbfile_read = openFile(self.filepath,'r')
        self.db_read = self.dbfile_read.root.frames
        self.schema_read = self.dbfile_read.root.schema
        
    def _append(self,frames):
        self._write_mode()
        # append the rows
        self.db_write.append(frames)
        self.db_write.flush()
        self._read_mode()
    
    def __len__(self):
        return self.db_read.__len__()
    
    def list_ids(self):
        l = self.db_read.readWhere('framen == 0')['_id']
        s = set(l)
        assert len(l) == len(s)
        return s
    
    def exists(self,source,external_id):
        rows = self.db_read.readWhere(\
                        '(source == "%s") & (external_id == "%s")' %\
                         (source,external_id))
        return len(rows) > 0
    
    def external_id(self,_id):
        row = self.db_read.readWhere('(_id == "%s") & (framen == 0)' % _id)
        return row[0]['source'],row[0]['external_id']
    
    def get_dtype(self,key):
        return getattr(self.db_read.cols,key).dtype
    
    def get_dim(self,key):
        return getattr(self.db_read.cols,key).shape[1:]
    
    
    
    def iter_feature(self,_id,feature):
        
        # BUG: The following should work, but always raises a
        # StopIteration exception around 30 - 40 rows. I have
        # no clue why.
        #
        # for row in self.db_read.where('_id == "%s"' % _id):
        #   yield row[feature]
        
        rowns = self.db_read.getWhereList('_id == "%s"' % _id)
        for row in self.db_read.itersequence(rowns):
            yield row[feature]
    
    @property
    def _temp_filepath(self):
        '''
        For use during a sync.  Return a modified version of the current
        filename, like 'frames_sync.h5', or something.
        '''
        fn,extension = os.path.splitext(self.filepath)
        return '%s_sync%s' % (fn,extension)
        
    
    
    def sync(self,add,update,delete,recompute):
        # each process needs its own reader
        newc = PyTablesFrameController(self.model,self._temp_filepath)
        new_ids = newc.list_ids()
        _ids = self.list_ids()
        for _id in _ids:
            if _id in new_ids:
                # this id has already been processed
                continue
            # This _id hasn't been inserted into the new PyTables file yet
            p = Pattern(_id,*self.external_id(_id))
            # create a transitional extractor chain that is able to read
            # features from the old database that don't need to be 
            # recomputed
            ec = self.model.extractor_chain(p,
                                            transitional=True,
                                            recompute = recompute)
            # process this pattern and insert it into the new database
            newc.append(ec)
        
        
        if (len(self) != len(newc)) or _ids != newc.list_ids():
            # Something went wrong. The number of rows or the set of _ids
            # don't match
            raise PyTablesUpdateNotCompleteError()
        
        # close both the new and old files
        newc.close()
        self.close()
        # remove the old file
        os.remove(self.filepath)
        # rename the temp file to the name of the old file
        os.rename(self._temp_filepath,self.filepath)
        # reload
        self._load(self.filepath)
    
    # TODO: Make sure there are tests
    def get_features(self):
        s = self.schema_read[:]['bytes'].tostring()
        return cPickle.loads(s)
        
    # TODO: This should return a Frames-derived instance
    def get(self,_id,features=None):
        return self.db_read.readWhere('_id == "%s"' % _id)
    
    def __getitem__(self,_id):
        return self.get(_id)
    
    def close(self):
        if self.dbfile_write:
            self.dbfile_write.close()
        if self.dbfile_read:
            self.dbfile_read.close()
    
    def __del__(self):
        self.close()
        
    
    
    
    
   
    
    
    
         
        

class DictFrameController(FrameController):
    
    def __init__(self,framesmodel):
        FrameController.__init__(self,framesmodel)
        
    
    
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
    
    def __len__(self):
        raise NotImplemented()
    
    def external_id(self,_id):
        raise NotImplemented()
    
    def list_ids(self):
        raise NotImplemented()
    
    def iter_feature(self,_id,feature_name):
        raise NotImplemented()
    
    def exists(self,source,external_id):
        raise NotImplemented()
        
        