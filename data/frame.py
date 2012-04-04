import string
import os.path
import re
import time
import cPickle
from abc import ABCMeta,abstractmethod

from tables import \
    openFile,IsDescription,StringCol,Col,Int8Col

import numpy as np

from celery.task import task,chord

from controller import Controller
from model.pattern import Pattern
import model.frame
from util import pad,ensure_path_exists

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
    def get(self,key):
        '''
        Gets rows, and optionally specific features from those rows.
        Indices may be a single index, a list of indices, or a slice.
        features may be a single feature or a list of them.
        '''
        pass
    
    
    def __getitem__(self,key):
        return self.get(key)
    
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
        Get the data type of the feature with key. Key may be a Feature, or a 
        string
        '''
        pass
    
    @abstractmethod
    def get_dim(self,key):
        '''
        Get the dimension of the feature with key. Key may be a feature, or a
        string
        '''
        pass
    
    @abstractmethod
    def iter_all(self,step = 1):
        '''
        Iterate over all frames, returning two-tuples of address,frames.  If 
        the step is greater than one, care should be taken to never return
        frames that span two patterns!
        '''
        pass

class PyTablesUpdateNotCompleteError(BaseException):
    '''
    Raised when a PyTables update fails
    '''
    def __init__(self):
        BaseException.__init__(self,Exception('The PyTables update failed'))



class PyTablesFrameController(FrameController):
    
    '''
    A FrameController that stores feature data in the hdf5 file format, and 
    uses the PyTables library to access it.
    
    PyTables has some special limitations, .e.g, columns cannot be added or
    removed after table creation. This class attempts to hide some of the messy
    details from clients
    
    Access by row numbers wins out speed-wise over access by indexed column 
    (e.g. where _id == 1234), so a slice with literal row numbers is the best
    address for this controller
    '''
    
    class Address(model.frame.Address):
        '''
        An address whose key can be anything acceptable
        for the tables.Table.__getitem__ method, i.e., an int,
        a slice, or a list of ints
        '''
        def __init__(self,key):
            if isinstance(key,int) or isinstance(key,np.integer):
                # address of a single frame
                self._len = 1
            elif isinstance(key,slice):
                if not (not key.step or 1 == key.step):
                    # step sizes of greater than one aren't allowed
                    raise ValueError(
                        'when using a slice as an address key,\
                         it must have a step of 1')
                self._len = key.stop - key.start
            elif isinstance(key,list) or isinstance(key,np.ndarray):
                # the address is a list of frame numbers, which may or may
                # not be contiguous
                self._len = len(key)
            else:
                raise ValueError(
                        'key must be an int, a list of ints, or a slice')
             
            model.frame.Address.__init__(self,key)
        
        def __str__(self):
            return '%s - %s' % (self.__class__,self.key)
        
        def serialize(self):
            raise NotImplemented()
        
        @classmethod
        def deserialize(cls):
            raise NotImplemented()
        
        def __len__(self):
            return self._len
    
    
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self,framesmodel)
        self._load(filepath)
    
    def _load(self,filepath):
        '''
        
        '''
        self.filepath = filepath
        self.filename = os.path.split(filepath)[-1]
        self.dbfile_write = None
        self.db_write = None
        self.dbfile_read = None
        self.db_read = None
        ensure_path_exists(self.filepath)
            
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
            self.dbfile_write.createTable(self.dbfile_write.root, 
                                          'frames',
                                           desc)
            self.db_write = self.dbfile_write.root.frames
            
            # create a table to store our schema as a pickled byte array
            class FrameSchema(IsDescription):
                bytes = Int8Col(pos = 0)
            
            self.dbfile_write.createTable(\
                        self.dbfile_write.root,'schema',FrameSchema)
            self.schema_write = self.dbfile_write.root.schema
            s = cPickle.dumps(\
                        self.model.features,cPickle.HIGHEST_PROTOCOL)
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
    
    def _append(self,frames):
        self._write_mode()
        # append the rows
        self.db_write.append(frames)
        self.db_write.flush()
        self._read_mode()
    
    def append(self,chain):
        bufsize = self._buffer_size
        rootkey = filter(lambda e : e.finite,chain)[0].key
        bucket = np.recarray(self._max_buffer_size,self.recarray_dtype)
        nframes = 0
        current = dict((k,0) for k in self.steps.keys())
        start_row = None
        for k,v in chain.process():
            if rootkey == k and \
                (nframes == bufsize or nframes >= self._max_buffer_size):
                
                # we've reached our smallest buffer size. Let's attempt a write
                try:
                    self.acquire_lock(nframes)
                    if start_row is None:
                        start_row = self.db_read.__len__()
                    # we got the lock. Let's write the data we have
                    self._append(bucket[:nframes])
                    nframes = 0
                    bucket[:] = 0
                    current = dict((k,0) for k in self.steps.keys())
                except PyTablesFrameController.WriteLockException:
                    # someone else has the write lock. Let's just keep 
                    # processing for awhile (within reason)
                    bufsize += self._buffer_size
                    
            if rootkey == k:
                nframes += 1
            try:
                steps = self.steps[k]
                cur = current[k]
                data = np.array(v).repeat(steps, axis = 0)
                bucket[k][cur:cur+steps] = data
                current[k] += steps
            except KeyError:
                # this feature isn't stored
                pass
            
            
        # We've processed the entire file. Wait until we can get the write lock    
        self.acquire_lock(nframes,wait=True)
        if start_row is None:
            start_row = self.db_read.__len__()
        
        print 'appending %i rows' % nframes
        self._append(bucket[:nframes])
        
        stop_row = self.db_read.__len__()
        # release the lock for the next writer
        self.release_lock()
        
         
        return PyTablesFrameController.Address(slice(start_row,stop_row))
    
        
    
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
        
    
    
    def __len__(self):
        return self.db_read.__len__()
    
    def list_ids(self):
        l = self.db_read.readWhere(self._query(framen = 0))['_id']
        s = set(l)
        assert len(l) == len(s)
        return s
    
    def exists(self,source,external_id):
        
        rows = self.db_read.readWhere(\
                    self._query(source = source,external_id = external_id))
        return len(rows) > 0
    
    def external_id(self,_id):
        
        row = self.db_read.readWhere(self._query(_id = _id, framen = 0))
        return row[0]['source'],row[0]['external_id']
    
    def get_dtype(self,key):
        if isinstance(key,model.frame.Feature):
            key = key.key
        return getattr(self.db_read.cols,key).dtype
    
    def get_dim(self,key):
        if isinstance(key,model.frame.Feature):
            key = key.key
        return getattr(self.db_read.cols,key).shape[1:]
    
    
    
    def iter_feature(self,_id,feature):
        
        # BUG: The following should work, but always raises a
        # StopIteration exception around 30 - 40 rows. I have
        # no clue why.
        #
        # for row in self.db_read.where('_id == "%s"' % _id):
        #   yield row[feature]
        
        # Here's the less simple workaround
        rowns = self.db_read.getWhereList(self._query(_id = _id))
        for row in self.db_read.itersequence(rowns):
            yield row[feature]
    
    def iter_all(self, step = 1):
        _ids = list(self.list_ids())
        for _id in _ids:
            rowns = self.db_read.getWhereList(self._query(_id = _id))
            last = rowns[-1]
            for rn in rowns[::step]:
                if 1 == step:
                    key = rn
                elif last - rn >= step:
                    key = slice(rn,rn + step,1)
                elif last == rn:
                    key = last
                else:
                    key = slice(rn,last,1)
                address = PyTablesFrameController.Address(key)
                yield  address,self.model[address]
    
    @property
    def _temp_filepath(self):
        '''
        For use during a sync.  Return a modified version of the current
        filename, like 'frames_sync.h5', or something.
        '''
        fn,extension = os.path.splitext(self.filepath)
        return '%s_sync%s' % (fn,extension)
        
    
    
            
    def sync(self,add,update,delete,recompute):
        
        if self.model.env().parallel:
            callback = sync_complete.subtask()
            _ids = self.list_ids()
            header = [sync_one.subtask(\
                        (self.model,self.filepath,_id,add,update,delete,recompute))\
                       for _id in _ids]
            # BUG: The callback is never called
            result = chord(header)(callback)
            result.get()
            self._load(self.filepath)
            return
        
        self._sync(add,update,delete,recompute)
    
    def _sync(self,add,update,delete,recompute):
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
    
    # TODO: Write tests
    def _query_condition(self,key,value):
        v = '"%s"' % value if isinstance(value,str) else str(value)
        return '(%s == %s)' % (key,v)
    
    def _query(self,op='&',**kwargs):
        return '(%s)' % string.join(\
            [self._query_condition(k,v) for k,v in kwargs.iteritems()],' %s ' % op)
        
    
    # TODO: Write tests
    def get(self,key):
        
        # the key is a zounds id
        if isinstance(key,str):
            return self.db_read.readWhere(self._query(_id = key))
        
        # the key is a (source, external id) pair
        if isinstance(key,tuple) \
            and 2 == len(key) \
            and all([isinstance(k,str) for k in key]):
            source,extid = key
            return self.db_read.readWhere(\
                            self._query(source = source, external_id = extid))
        
        # key is an address, which means that it's an int, a slice, or a list
        # of ints, all of which can be used to address a tables.Table instance
        # directly
        if isinstance(key,PyTablesFrameController.Address):
            return self.db_read[key.key]
        
        raise ValueError(\
            'key must be a zounds id, a (source,external_id) pair,\
             or a PyTablesFrameController.Address instance')
    
    
    def close(self):
        if self.dbfile_write:
            self.dbfile_write.close()
        if self.dbfile_read:
            self.dbfile_read.close()
    
    def __del__(self):
        self.close()
        
    
    
    
# KLUDGE: this should be a PyTablesFrameController class method, if at all possible
@task(name='data.frame.sync_one')
def sync_one(newmodel,filepath,_id,add,update,delete,recompute):
    '''
    '''
    oldc = PyTablesFrameController(newmodel,filepath)
    newc = PyTablesFrameController(newmodel,oldc._temp_filepath)
    _id_query = '_id == "%s"' % _id
    oldrows = oldc.db_read.getWhereList(_id_query)
    newrows = newc.db_read.getWhereList(_id_query)
    oldlen = len(oldrows)
    newlen = len(newrows)
    
    if oldlen == newlen:
        # this id has already been processed
        return
     
    if newlen:
        # There are some rows in the new database with id, but
        # there aren't the same number as oldlen, meaning that 
        # something probably went wrong during the sync. To keep
        # things simple, let's delete what's there and start over
        newc.acquire_lock(newc._max_buffer_size, wait = True)
        newc._write_mode()
        newc.db_write.removeRows(newrows)
        newc._read_mode()
        newc.release_lock()
    
    p = Pattern(_id,*oldc.external_id(_id))
    ec = newmodel.extractor_chain(p,transitional = True,recompute = recompute)
    newc.append(ec)
    print 'processed %s' % _id
    oldc.close()
    newc.close()
    return (newmodel,filepath)
    

# KLUDGE: this should be a PyTablesFrameController class method, if at all possible
@task(name='data.frame.sync_complete')
def sync_complete(results):
    newmodel = results[0][0]
    filepath = results[0][1]
    oldc = PyTablesFrameController(newmodel,filepath)
    tmpfilepath = oldc._temp_filepath
    newc = PyTablesFrameController(newmodel,tmpfilepath)
    oldids = oldc.list_ids()
    newids = newc.list_ids()
    
    if (len(oldc) != len(newc) or oldids != newids):
        raise PyTablesUpdateNotCompleteError()
    
    
    oldc.close()
    newc.close()
    os.remove(filepath)
    os.rename(tmpfilepath,filepath)
    print 'sync complete'
    return True
   
    
    
    
         
        

class DictFrameController(FrameController):
    '''
    Completely useless, except for testing
    '''
    
    def __init__(self,framesmodel):
        FrameController.__init__(self,framesmodel)
        
    def sync(self,add,update,delete,chain):
        raise NotImplemented()
    
    def append(self,frames):
        raise NotImplemented()
    
    def get(self,key):
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
    
    def iter_all(self):
        raise NotImplemented()
        
        
