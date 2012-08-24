import string
import os.path
import re
import time
import cPickle

from tables import \
    openFile,IsDescription,StringCol,Col,Int8Col

import numpy as np
import zounds.model.frame
from zounds.constants import audio_key,id_key,source_key,external_id_key
from zounds.model.pattern import Pattern
from zounds.util import ensure_path_exists
from frame import FrameController

class PyTablesFrameController(FrameController):
    
    '''
    A FrameController that stores feature data in the hdf5 file format, and 
    uses the PyTables library to access it.
    
    PyTables has some special limitations, .e.g, columns cannot be added or
    removed after table creation. This class attempts to hide some of the messy
    details from clients
    
    Access by row numbers wins out speed-wise over access by indexed column 
    (e.g. where _id == 1234), so a slice with literal row numbers is the best
    address for this controller.
    
    This class takes for granted that only addresses from the same pattern
    will be compared.  
    '''
    
    class Address(zounds.model.frame.Address):
        
        # key types
        INTEGER = object()
        SLICE = object()
        NPARR = object()
        
        '''
        An address whose key can be anything acceptable
        for the tables.Table.__getitem__ method, i.e., an int,
        a slice, or a list of ints
        '''
        def __init__(self,key):
            zounds.model.frame.Address.__init__(self,key)
            if isinstance(key,int) or isinstance(key,np.integer):
                # address of a single frame
                self._len = 1
                self._key_type = PyTablesFrameController.Address.INTEGER
                self.min = key
                self.max = key
            elif isinstance(key,slice):
                if not (not key.step or 1 == key.step):
                    # step sizes of greater than one aren't allowed
                    raise ValueError(
                        'when using a slice as an address key,\
                         it must have a step of 1')
                self._len = key.stop - key.start
                self._key_type = PyTablesFrameController.Address.SLICE
                self.min = key.start
                self.max = key.stop - 1
            elif isinstance(key,list):
                # the address is a list of frame numbers, which may or may
                # not be contiguous
                self._len = len(key)
                self._key_type = PyTablesFrameController.Address.NPARR
                key = np.array(key,dtype=np.int32)
                self.min = key.min()
                self.max = key.max()
            elif isinstance(key,np.ndarray):
                # the address is a list of frame numbers, which may or may
                # not be contiguous
                self._len = len(key)
                self._key_type = PyTablesFrameController.Address.NPARR
                self.min = key.min()
                self.max = key.max()
            else:
                raise ValueError(
                        'key must be an int, a list of ints, or a slice')
            
            self._span = self.max - self.min
            
        
        def __str__(self):
            return '%s - %s' % (self.__class__,self.key)
        
        def __repr__(self):
            return self.__str__()
        
        def serialize(self):
            raise NotImplemented()
        
        @classmethod
        def deserialize(cls):
            raise NotImplemented()
        
        def __len__(self):
            return self._len
        
        @property
        def span(self):
            return self._span
        
        def __eq__(self,other):
            if len(self) != len(other):
                return False
            
            if self.span != other.span:
                return False
            
            return self.min == other.min and self.max == other.max
        
        def __hash__(self):
            return hash((self._len,self.min,self.max))
            
        def __ne__(self,other):
            return not self.__eq__(other)
        
        def __lt__(self,other):
            return self.min < other.min
        
        def __le__(self,other):
            return self.__lt__(other) or self.__eq__(other)
        
        def __gt__(self,other):
            return self.min > other.min
        
        def __ge__(self,other):
            return self.__gt__(other) or self.__eq__(other)
    
        # TODO: Write tests
        @classmethod
        def congeal(cls,addresses):
            if None is addresses or not len(addresses):
                raise ValueError(\
                    'addresses must be an sequence with length greater than 0')
            if 1 == len(addresses):
                return addresses[0]
            
            srt = sorted(addresses)
            return cls(slice(srt[0].min,srt[-1].max))
                    
    
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
            # Don't update indexes automatically. This significantly slows down
            # write times, because the index is updated each time flush() is called.
            # This means write times get slower and slower as the table size
            # increases.
            self.db_write.autoIndex = False
            
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
                if (isinstance(col,StringCol) or oned):
                    try:
                        col.createIndex()
                    except NotImplementedError:
                        # Indexing of unsigned 64 bit columns is not currently
                        # supported in PyTables 2.3.1
                        pass
                    
            self.dbfile_write.close()
            

        self.dbfile_read = openFile(filepath,'r')
        self.db_read = self.dbfile_read.root.frames
        self.schema_read = self.dbfile_read.root.schema
                
        self.recarray_dtype = []
        for k in self.db_read.colnames:
            col = getattr(self.db_read.cols,k)
            self.recarray_dtype.append((k,col.dtype,col.shape[1:]))
         
        self.has_lock = False
        # Between the time append is first called, and the database is re-indexed,
        # we're going to keep an in memory list of external ids.  This is because
        # we don't want to update the indexes on each flush() call, but acquirers
        # need to check if a particular (source,external_id) pair already exists
        # in the database.
        self._temp_external_ids = None
        # update the indexes, if need be
        self.update_index()
    
    @property
    def concurrent_reads_ok(self):
        return False
    
    @property
    def concurrent_writes_ok(self):
        return False
        
    def update_index(self):
        '''
        Force updates on all table indexes.
        '''
        self._write_mode()
        self.db_write.reIndexDirty()
        # the indexes have been updated, so external_ids queries will be fast
        # again. Throw away the in-memory store.
        self._temp_external_ids = None
        self._read_mode()
    
    def _append(self,frames):
        self._write_mode()
        # append the rows
        self.db_write.append(frames)
        self.db_write.flush()
        self._read_mode()
    
    def _ensure_lock_and_append(self,record):
        self.acquire_lock()
        self._append(record)
        self.release_lock()
    
    
    def append(self,chain):
        
        def safe_concat(a,b):
            if None is a:
                return b
            return np.concatenate([a,b])
        
        rootkey = audio_key
        start_row = None
        # Wait to initialize the abs_steps values, since the Precomputed
        # extractor doesn't know its step until the first call to _process()
        abs_steps = dict([(e.key,None) for e in chain])
        data = dict([(k,None) for k in self.steps.iterkeys()])
        chunks_processed = 0
        for k,v in chain.process():
            if k not in self.steps:
                continue
    
            if None is abs_steps[k]:
                # The first call to _process has been made, so it's safe to
                # get the absolute step value for this extractor
                abs_steps[k] = chain[k].step_abs()
            
            if rootkey == k and chunks_processed > 0:
                if None is start_row:
                    start_row = self.db_read.__len__()
                record = self._recarray(rootkey, data, done = False)
                self._ensure_lock_and_append(record)
                
            
            f = np.repeat(v, abs_steps[k], 0)
            data[k] = safe_concat(data[k],f)
            if rootkey == k:
                chunks_processed += 1
        
        if None is start_row:
            start_row = self.db_read.__len__()
        record = self._recarray(rootkey, data, done = True)
        self._ensure_lock_and_append(record)
    

        stop_row = self.db_read.__len__()
        if self._temp_external_ids is None:
            # This is the first time append() has been called since the indexes
            # have been updated. Create the in-memory list of external_ids
            self._temp_external_ids = \
                dict(((t,None) for t in self.list_external_ids()))
        
        row = record[0]
        source = row[source_key]
        external_id = row[external_id_key]
        # update the in-memory hashtable of external ids
        self._temp_external_ids[(source,external_id)] = None
        return PyTablesFrameController.Address(slice(start_row,stop_row))
        
        
    @property
    def lock_filename(self):
        return self.filepath + '.lock'
    
    class WriteLockException(BaseException):
        
        def __init__(self):
            BaseException.__init__(self)
    
    def acquire_lock(self):
        if self.has_lock:
            return
        
        locked = os.path.exists(self.lock_filename)
        while locked:
            time.sleep(0.1)
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
    
    def address(self,_id):
        rowns = self.db_read.getWhereList(self._query(_id = _id))
        return PyTablesFrameController.Address(slice(rowns[0],rowns[-1] + 1))
    
    def list_ids(self):
        l = self.db_read.readWhere(self._query(framen = 0))['_id']
        s = set(l)
        assert len(l) == len(s)
        return s
    
    def list_external_ids(self):
        rows = self.db_read.readWhere(self._query(framen = 0))
        return zip(rows[source_key],rows[external_id_key])
    
    def exists(self,source,external_id):
        if self._temp_external_ids is not None:
            # we've currently got an in-memory hashtable of external_ids that's
            # being kept up-to-date. It's safe, and much faster to use it.
            return (source,external_id) in self._temp_external_ids
        
        # We need to query the database directly.
        rows = self.db_read.readWhere(\
                    self._query(source = source,external_id = external_id))
        return len(rows) > 0
    
    def external_id(self,_id):
        
        row = self.db_read.readWhere(self._query(_id = _id, framen = 0))
        r = row[0]
        return r[source_key],r[external_id_key]
    
    def get_dtype(self,key):
        key = self._feature_as_string(key)
        return getattr(self.db_read.cols,key).dtype
    
    def get_dim(self,key):
        key = self._feature_as_string(key)
        return getattr(self.db_read.cols,key).shape[1:]
    
    
    def stat(self,feature,aggregate,axis = 0, step = 1):
        key = feature if isinstance(feature,str) else feature.key
        return aggregate([row[key] for row in self.db_read[::step]],axis = axis)
    
    def iter_feature(self,_id,feature,step = 1,chunksize = 1):
        feature = feature if isinstance(feature,str) else feature.key
        rowns = self.db_read.getWhereList(self._query(_id = _id))[::step]
        if chunksize == 1:
            for row in self.db_read.itersequence(rowns):
                yield row[feature]
        else:
            # iterate from the first row number to 1 plus the last row number,
            # since the row numbers are inclusive
            for i in xrange(rowns[0],rowns[-1] + 1,chunksize):
                stop = i + chunksize
                indices = np.where((rowns >= i) & (rowns < stop))[0]
                yield self.db_read[rowns[indices]][feature]
    
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
                
    
    def iter_id(self,_id,chunksize,step = 1):
        # get all the row numbers occupied by this id
        rowns = self.db_read.getWhereList(self._query(_id = _id))
        # convert the row numbers to an address instance
        try:
            address = PyTablesFrameController.Address(slice(rowns[0],rowns[-1] + 1))
        except IndexError:
            return
        
        frames = self.model[address]
        
        for i in xrange(0,len(rowns),step):
            if i + chunksize >= len(rowns) - 1:
                break
            rns = rowns[i:i+chunksize:step]
            fsl = np.array(range(i,i + chunksize,step))
            yield PyTablesFrameController.Address(slice(rns[0],rns[-1])), frames[fsl]

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
            print 'updating %s - %s' % (p.source,p.external_id)
            # process this pattern and insert it into the new database
            newc.append(ec)
        
        
        if (len(self) != len(newc)) or _ids != newc.list_ids():
            # Something went wrong. The number of rows or the set of _ids
            # don't match
            raise UpdateNotCompleteError()
        
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
    
    def get(self,key):
        
        # the key is a zounds id
        if isinstance(key,str):
            return self.db_read.readWhere(self._query(_id = key))
        
        # the key is a (source, external id) pair
        if isinstance(key,tuple) \
            and 2 == len(key) \
            and all([isinstance(k,str) for k in key]):
            
            source,extid = key
            print source,extid
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
        

# Crazy Bad KLUDGE: I rely on FrameController-derived classes to define a back-end
# specific Address class as a nested class. This makes those classes impossible
# to pickle, however.  This is a baaad solution, but I'd like to keep moving
# for right now.    
import sys         
setattr(sys.modules[__name__], 'Address', PyTablesFrameController.Address)