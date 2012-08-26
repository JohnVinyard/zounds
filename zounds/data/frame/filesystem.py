from __future__ import division
import os
import shutil
import cPickle
from itertools import repeat
import numpy as np

import zounds.model.frame
from zounds.constants import audio_key,id_key,source_key,external_id_key
from frame import FrameController,UpdateNotCompleteError
from zounds.model.pattern import Pattern
from zounds.util import ensure_path_exists
from time import time,sleep

'''
Directory Structure

DBNAME
    - features.dat
    - external_ids.dat
    - lengths.dat
    - sync.dat?
    /DATA
        - ZoundsId.dat
        ...
'''

# TODO: Consider adding some read methods as well
class DataFile(object):
    
    def __init__(self,controller):
        self._file = None
        self._id = None
        self._source = None
        self._external_id = None
        self._c = controller
        
    
    def write(self,_id,source,external_id,data):
        if None is self._file:
            self._id = _id
            self._source = source
            self._external_id = external_id
            self._file = open(self._c._pattern_path(_id),'wb')
            source = np.array(source,dtype = 'a%i' % self._c._source_chars)
            external_id = np.array(\
                            external_id,dtype = 'a%i' % self._c._extid_chars)
            self._file.write(source)
            self._file.write(external_id)
        
        self._file.write(data.tostring())
        return len(data)
    
    def close(self):
        self._file.close()

class ConcurrentIndex(object):
    
    def __init__(self,builders,indexpath,datadir,iterator):
        object.__init__(self)
        self._keys = ['_id','external_id','length']
        self._builders = builders
        self._path = os.path.join(indexpath,'index.dat')
        self._lock_filename = os.path.join(indexpath,'lock.dat')
        self._datadir = datadir
        self._iterator = iterator
        self._in_mem_timestamp = None
        self._data = dict([(k,dict()) for k in self._keys])
        self.startup()
    
    def __getitem__(self,k):
        # Check if memory is stale, if not, return. if so:
        # Check if file is stale, if not, load and return, if so:
        # reindex, save, and return
        if not self.memory_is_stale():
            # The in-memory index is up-to-date
            return self._data[k]
        
        if not self.file_is_stale():
            # The in-memory index was stale, but the file on disk is current.
            # Just load it up and return the value
            self.wait_for_unlock()
            self._data = self.from_disk()
            self._in_mem_timestamp = time()
            return self._data[k]
        
        # Both the in-memory and on-disk indexes are stale, rebuild the index,
        # persist it to disk, and return the requested value.
        self.acquire_lock()
        self.reindex()
        self.to_disk()
        self.release_lock()
        self._in_mem_timestamp = time()
        return self._data[k]
    
    def startup(self):
        if self.ensure_condition(self.file_is_stale):
            self.reindex()
            self.to_disk()
            self._in_mem_timestamp = time()
            self.release_lock()
        else:
            self.wait_for_unlock()
            self._data = self.from_disk()
            self._in_mem_timestamp = time()
    
    def append(self,_id,source,external_id,length):
        self.acquire_lock()
        if self.memory_is_stale():
            # another controller has updated the disk index. Our in-memory
            # version is out-of-date. Merge the disk version with our version
            # before doing anything else.
            fresh = self.from_disk()
            for k in fresh.iterkeys():
                self._data[k].update(fresh[k])
        
        self._data['_id'][_id] = (source,external_id)
        self._data['external_id'][(source,external_id)] = _id
        self._data['length'][_id] = length
        self.to_disk()
        self._in_mem_timestamp = time()
        self.release_lock()
    
    def ensure_condition(self,c):
        '''
        Parameters
            c - a callable, taking no parameters, that returns a boolean value
        Returns
            True if the callable returns true both before and after acquiring a
            lock, otherwise False.
        '''
        b = c()
        if not b:
            return False
        self.acquire_lock()
        return c()
        
    
    def from_disk(self):
        with open(self._path,'r') as f:
            return cPickle.load(f)
    
    def to_disk(self):
        with open(self._path,'w') as f:
            cPickle.dump(self._data,f)
    
    def reindex(self):
        for metadata in self._iterator(self._data['_id']):
            for index_key in self._keys:
                builder = self._builders[index_key]
                k,v = builder(*metadata)
                self._data[index_key][k] = v

    
    def id_from_external_id(self,extid):
        return self._data['external_id'][extid]
    
    def external_id_from_id(self,_id):
        return self._data['_id'][_id]
    
    def pattern_length(self,_id):
        return self._data['length'][_id]
    
    @property
    def is_locked(self):
        return os.path.exists(self._lock_filename)
    
    def make_lock(self):
        f = open(self._lock_filename,'w')
        f.close()
    
    def acquire_lock(self):
        self.wait_for_unlock()
        self.make_lock()
    
    def release_lock(self):
        os.remove(self._lock_filename)
    
    def wait_for_unlock(self):
        while self.is_locked:
            sleep(0.05)
    
    
    def file_is_stale(self):
        '''
        Returns true if pattern data has been added to the data directory more
        recently than the index file has been updated
        '''
        return (not os.path.exists(self._path)) or \
                (os.path.getmtime(self._datadir) > os.path.getmtime(self._path))
    
    
    def memory_is_stale(self):
        '''
        Returns true if another ConcurrentIndex instance has updated the persistent
        index.  Our in-memory version needs to be refreshed.
        '''
        return os.path.getmtime(self._path) > self._in_mem_timestamp
    
        

class FileSystemFrameController(FrameController):
    
    # KLUDGE: As it is, this class isn't able to handle addresses that contain
    # frames from multiple patterns. An address *should* be able to address any
    # number of patterns, with frames in any order.
    #
    # TODO: What can I factor out from this and PyTablesFrameController.Address?    
    class Address(zounds.model.frame.Address):
        '''
        Address is a two-tuple of (_id,int,iterable of ints, or slice)
        '''
        def __init__(self,key):
            zounds.model.frame.Address.__init__(self,key)
            self._id,self._index = self.key
            try:
                # try to treat index as an iterable of ints
                self._len = len(self._index)
                self._index = np.array(self._index)
                self.min = self._index.min()
                self.max = self._index.max()
                return
            except TypeError:
                pass
            
            try:
                # try to treat index as a slice
                self._len = self._index.stop - self._index.start
                self.min = self._index.start
                self.max = self._index.stop - 1
                return
            except AttributeError:
                pass
            
            if isinstance(self._index,int):
                # The index must be a single int
                self._len = 1
                self.min = self._index
                self.max = self._index
            else:
                raise ValueError(\
                    'key must be a two-tuple of (_id,int, iterable of ints, or slice)')
            
        
        @property
        def _span(self):
            return self.max - self.min

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
            if not self.__ideq__(other):
                return False
            
            if len(self) != len(other):
                return False
            
            if self.span != other.span:
                return False
            
            return self.min == other.min and self.max == other.max
        
        def __ideq__(self,other):
            return self._id == other._id
        
        def __hash__(self):
            return hash((self._id,self._len,self.min,self.max))
        
        def __ne__(self,other):
            return not self.__eq__(other)
        
        def _check(self,other):
            if not self.__ideq__(other):
                raise ValueError(\
                'Comparisons are meaningless for addresses with different ids')
        
        def __lt__(self,other):
            self._check(other)
            return self.min < other.min
        
        def __le__(self,other):
            self._check(other)
            return self.__lt__(other) or self.__eq__(other)
        
        def __gt__(self,other):
            self._check(other)
            return self.min > other.min
        
        def __ge__(self,other):
            self._check(other)
            return self.__gt__(other) or self.__eq__(other)
        
        @classmethod
        def congeal(cls,addresses):
            if None is addresses or not len(addresses):
                raise ValueError(\
                    'addresses must be an sequence with length greater than 0')
            if 1 == len(addresses):
                return addresses[0]
            srt = sorted(addresses)
            return cls((srt[0]._id,slice(srt[0].min,srt[-1].max)))
        
    
    file_extension = '.dat'
    features_fn = 'features' + file_extension
    sync_fn = 'sync' + file_extension
    data_dn = 'data'
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self,framesmodel)
        self._source_chars = 32
        self._extid_chars = 32
        self._metadata_chars = self._source_chars + self._extid_chars
        self._metadata_fmt = '%ic' % self._metadata_chars
        self._load(filepath)
        
    
    def _load(self,filepath):
        self._rootdir = filepath
        
        self._datadir = \
            os.path.join(self._rootdir,FileSystemFrameController.data_dn)
        
        ensure_path_exists(self._datadir)
        fsfc = FileSystemFrameController
        self._feature_path = os.path.join(self._rootdir,fsfc.features_fn)
        
        self._sync_path = os.path.join(self._rootdir,fsfc.sync_fn)
        self._load_features()
        
        # Get the keys and shapes of all stored features
        dims = self._dimensions
        self._dtype = [(k,v[1],v[0]) for k,v in dims.iteritems()]
        self._np_dtype = np.dtype(self._dtype)
        
        
        # Features that are redundant (i.e., the same for every frame), and
        # won't be stored on disk as part of the recarray
        self._excluded_metadata = [id_key,source_key,external_id_key]
        # Only the feature keys that will be stored, and aren't redundant metadata
        self._skinny_features = filter(\
            lambda a : a not in self._excluded_metadata,dims.iterkeys())
        # The dtype of the recarrays that will be stored on disk
        self._skinny_dtype = np.dtype(filter(\
            lambda a : a[0] in self._skinny_features,
            self._dtype))
        
        self._reindex_if_necessary()
    
    
    def _reindex_if_necessary(self):
        builders = {'_id' : self._build_id_index,
                    'external_id' : self._build_external_id_index,
                    'length' : self._build_lengths_index}
        self._index = ConcurrentIndex(\
                    builders,self._rootdir,self._datadir,self._iter_patterns)
    
    def _fsfc_address(self,key):
        return FileSystemFrameController.Address(key)
    
    
    # TODO: Try np.lib.recfunctions.drop_fields
    def _to_skinny(self,record):
        # return only the fields of the record that will be stored on disk
        return record[self._skinny_features]
    
    # TODO: It would be much more efficient to write a wrapper around recarray
    # whose constructor takes literal fields. 
    # record[0]['literal'] would always return the same value
    # record['literal'] would return an array of the literal value the same length
    # as the "real" recarray. I'm not exactly sure how this would work with
    # multi-id addresses, however.
    #
    #
    # TODO: Try np.lib.recfunctions.append_fields
    def _from_skinny(self,_id,source,external_id,skinny_record):
        r = np.recarray(skinny_record.shape,self._np_dtype)
        r[id_key] = _id
        r[source_key] = source
        r[external_id_key] = external_id
        for sf in self._skinny_features:
            r[sf] = skinny_record[sf]
        return r

    def _recarray(self,rootkey,data,done = False):
        meta,record = FrameController._recarray(\
                self, rootkey, data, done = done, 
                dtype = self._skinny_dtype, meta = self._excluded_metadata)
        _id,source,external_id = meta
        return _id,source,external_id,record
    
    def _str_dtype(self,nchars):
        return 'a%i' % nchars
    
    def _get_memmap(self,addr):
        _id = addr._id
        slce = addr._index
        # path to the file containing data at addr
        fn = self._pattern_path(_id)
        # the size of the file, in bytes
        fs = os.path.getsize(fn)
        # the size of each record, in bytes
        size = self._skinny_dtype.itemsize
        # the total number of rows in the file
        nrecords = (fs - self._metadata_chars) / size
        # the start index, in rows
        start_index = 0 if None is slce.start else slce.start
        # the stop index, in rows
        stop_index = nrecords if None is slce.stop else slce.stop
        step = 1 if None is slce.step else slce.step
        # the number of rows to read
        to_read = stop_index - start_index
        with open(fn,'rb') as f:
            # read the metadata from the file
            source,extid = self._get_source_and_external_id(f)
            # get the offset, in bytes, where we'll start reading
            start_bytes = start_index * size 
            offset = start_bytes + self._metadata_chars
            data = np.memmap(\
                    f,dtype = self._skinny_dtype, mode = 'r',offset = offset)
        return _id,source,extid,data[:to_read:step]
    
    def _get_source_and_external_id(self,f):
        meta = f.read(self._metadata_chars)
        source = np.fromstring(meta[:self._source_chars],
                                   dtype = self._str_dtype(self._source_chars))[0]
        extid = np.fromstring(meta[self._source_chars:],
                                  dtype = self._str_dtype(self._extid_chars))[0]
        return source,extid
    
    def _get_hydrated(self,addr):
        return self._from_skinny(*self._get_memmap(addr))
        
    
    def _pattern_path(self,_id):
        return os.path.join(self._datadir,
                            _id + FileSystemFrameController.file_extension)
    
    def _file_is_stale(self,path):
        '''
        Compare an index file with the modified time of the data directory.  
        Return True if the data directory has been modified more recently than
        the index file.
        '''
        return os.path.getmtime(path) < os.path.getmtime(self._datadir)
    
    def _load_features(self):
        '''
        Ensure that the features of the current FrameModel are stored on disk
        '''
        try:
            with open(self._feature_path,'r') as f:
                self._features,self._dimensions = cPickle.load(f)
        except IOError:
            self._features = self.model.features
            self._dimensions = self.model.dimensions()
            with open(self._feature_path,'w') as f:
                cPickle.dump((self._features,self._dimensions),f)        
        
        
    def _build_id_index(self,_id,source,external_id,f):
        return _id,(source,external_id)
    
    def _build_external_id_index(self,_id,source,external_id,f):
        return (source,external_id),_id    
    
    def _build_lengths_index(self,_id,source,external_id,f):
        fs = os.path.getsize(os.path.abspath(f.name))
        nframes = (fs - self._metadata_chars) / self._skinny_dtype.itemsize
        return _id,nframes
        
    def _iter_patterns(self,d):
        files = os.listdir(self._datadir)
        for f in files:
            _id = f.split('.')[0]
            if _id in d:
                continue
            path = os.path.join(self._datadir,f)
            with open(path,'r') as of:
                source,external_id = self._get_source_and_external_id(of)
                yield _id,source,external_id,of
    
    @property
    def concurrent_reads_ok(self):
        return True
    
    @property
    def concurrent_writes_ok(self):
        return True
    
    @property
    def _ids(self):
        return self._index['_id']
    
    @property
    def _external_ids(self):
        return self._index['external_id']
    
    @property
    def _lengths(self):
        return self._index['length']
    
    # TODO: Detect directory changes (from another process or thread) and update
    # index if necessary
    def __len__(self):
        '''
        Keep nframes in the file metadata. On startup, count the frames. Keep
        track of frames in-memory as append() is called
        
        OR
        
        A mapreduce kinda-thing, which gets triggered to update when new files
        are added. This means that we avoid the startup cost, but have some
        extra complexity. The total length is stored in its own file.
        '''
        return np.sum(self._lengths.values())
    
    # TODO: Detect directory changes (from another process or thread) and update
    # index if necessary
    def list_ids(self):
        '''
        Read files from the directory where data is stored. Keep it in memory. Update
        the in-memory list as append() is called
        '''
        return set(self._ids.keys())
    
    # TODO: Detect directory changes (from another process or thread) and update
    # index if necessary
    def list_external_ids(self):
        '''
        External ids (source,external_id), will be stored in file metadata, just
        like frame lengths, so
        
        On startup, get a list of external_ids. Keep track of new ones as append()
        is called
        
        OR
        
        A mapreduce kinda-thing, which gets triggered when new files are added.
        The list of external_ids is stored in its own file.
        
        OR
        
        I may need to keep an in-memory external_id -> _id mapping
        '''
        return self._external_ids.keys()
    
    # TODO: Detect directory changes (from another process or thread) and update
    # index if necessary
    def external_id(self,_id):
        '''
        Read the external_id from the file's metadata and cache it.
        '''
        return self._ids[_id]
    
    # TODO: Detect directory changes (from another process or thread) and update
    # index if necessary
    def exists(self,source,external_id):
        '''
        Return true if the source,external_id pair is in the in-memory 
        external_id -> _id mapping
        '''
        return (source,external_id) in self._external_ids
    
    @property
    def _temp_path(self):
        '''
        For use during a sync. Return a modified version of the current path.
        '''
        return '%s_sync' % self._rootdir
    
    # TODO: A lot of this code is identical to what's in PyTablesFrameController.sync()
    def sync(self,add,update,delete,recompute):
        '''
        Create a new directory with a temporary name
        '''
        newc = FileSystemFrameController(self.model,self._temp_path)
        new_ids = newc.list_ids()
        _ids = self.list_ids()
        for _id in _ids:
            if _id in new_ids:
                # this id has already been processed
                continue
            p = Pattern(_id,*self.external_id(_id))
            ec = self.model.extractor_chain(\
                            p,transitional = True,recompute = recompute)
            print 'updating %s - %s' % (p.source,p.external_id)
            newc.append(ec)
            # this pattern has been successfully updated. It's safe to remove
            # it from the original db.
            os.remove(self._pattern_path(_id))
        
        if (len(self) != len(newc)) or _ids != newc.list_ids():
            # Something went wrong. The number of rows or the set of _ids
            # don't match
            raise UpdateNotCompleteError()
        
       
        # remove the old file
        shutil.rmtree(self._rootdir)
        # rename the temp file to the name of the old file
        os.rename(self._temp_path,self._rootdir)
        # reload
        # TODO: Consider making the _load function an abstract method on the
        # base class.
        self._load(self._rootdir)
    
    # TODO: A lot of this code is identical to PyTablesFrameController.sync().
    # Refactor!
    def append(self,chain):
        '''
        Create a new file. Add meta and real data
        '''
        def safe_concat(a,b):
            return b if None is a else np.concatenate([a,b])
        
        
        rootkey = audio_key
        # Wait to initialize the abs_steps values, since the Precomputed
        # extractor doesn't know its step until the first call to _process()
        abs_steps = dict([(e.key,None) for e in chain])
        data = dict([(k,None) for k in self.get_features().iterkeys()])
        data[audio_key] = None
        chunks_processed = 0
        nframes = 0
        datafile = DataFile(self)
        for k,v in chain.process():
            if k not in data:
                continue
    
            if None is abs_steps[k]:
                # The first call to _process has been made, so it's safe to
                # get the absolute step value for this extractor
                abs_steps[k] = chain[k].step_abs()
            
            if rootkey == k and chunks_processed > 0:
                frames = datafile.write(\
                            *self._recarray(rootkey, data, done = False))
                nframes += frames
                
            
            f = np.repeat(v, abs_steps[k], 0)
            data[k] = safe_concat(data[k],f)
            if rootkey == k:
                chunks_processed += 1
        
        frames = datafile.write(*self._recarray(rootkey, data, done = True))
        nframes += frames
        
        # update indexes
        self._index.append(datafile._id, datafile._source, datafile._external_id, nframes)
        addr = self._fsfc_address((datafile._id,slice(0,nframes)))
        datafile.close()
        return addr
    
    def address(self,_id):
        '''
        Return the address for an _id
        '''
        return self._fsfc_address((_id,slice(0,self._lengths[_id])))
    
    def get(self,key):
        '''
        Zounds ID - Fetch all the data from the file with name ID
        
        (Source,External Id) Lookup the Zounds ID from the in-memory hashtable.
          Load the file.
          
        Address (Zounds ID,Offset,Length) - Open a memmapped version of the file,
        Read only the frames specified
        '''
        
        _id = None
        if isinstance(key,str):
            # the key is a zounds id
            addr = self.address(key)
        elif isinstance(key,tuple) \
            and 2 == len(key) \
            and all([isinstance(k,str) for k in key]):
            # the key is a (source, external id) pair
            _id = self._external_ids[key]
            addr = self.address(_id)
        elif isinstance(key,self.address_class):
            addr = key
        else:
            raise ValueError('key must be a zounds id, a (source,external_id) pair,\
                            or a FileSystemFrameController.Address instance')
        
        return self._get_hydrated(addr)
        
    
    def stat(self,feature,aggregate,axis = 0,step = 1):
        '''
        Streaming, multiprocess min,max,sum,mean,std implementations.
        
        Store the results for each pattern in a companion file.
        
        Store the final result in a file.
        
        If the data directory's modified date is greater than the total modified
        date, recompute by using companion files or computing for new files.
        '''
        raise NotImplemented()
        
    
    def get_features(self):
        '''
        Return a dictionary mapping Feature Key -> Feature for the current FrameModel
        '''
        return self._features
    
    def get_dtype(self,key):
        '''
        Return the numpy datatype of the feature with key
        '''
        key = self._feature_as_string(key)
        dtype = self._np_dtype[key]
        try:
            return dtype.subdtype[0]
        except TypeError:
            return dtype
    
    def get_dim(self,key):
        '''
        Return the dimension of the feature with key
        '''
        key = self._feature_as_string(key)
        dtype = self._np_dtype[key]
        try:
            return dtype.subdtype[1]
        except TypeError:
            return ()
    
    def iter_all(self,step = 1):
        '''
        Iterate over all patterns, returning two-tuples of (Address,frames)
        '''
        _ids = self.list_ids()
        for _id in _ids:
            _id,source,external_id,mmp = self._get_memmap(self.address(_id))
            lmmp = len(mmp)
            for i in xrange(0,lmmp,step):
                key = i if step == 1 else slice(i,min(i + step,lmmp))
                addr = self._fsfc_address((_id,key))
                data = self._from_skinny(_id, source, external_id, mmp[key])
                yield addr,self.model(data = data)

    def iter_feature(self,_id,feature,step = 1, chunksize = 1):
        '''
        For file with Zounds ID, iterate over the feature in the manner specified.
        TODO: What does the underlying implementation of memmap[::step] do?
        '''
        # for metadata, return an iterator that outputs the correct value as
        # many times as necessary
        
        # for non-metdata, iterate over a memmapped file.
        feature = self._feature_as_string(feature)
        l = self._lengths[_id]
        _id,source,external_id,mmp = self._get_memmap(\
                self._fsfc_address((_id,slice(0,l,step))))
        
        meta = feature in self._excluded_metadata
        if meta:
            rp = repeat(locals()[feature],len(mmp))
            for r in rp:
                yield r 
        else:
            if 1 == chunksize:
                for row in mmp:
                    yield row[feature]
            else:
                chunks_per_step = int(chunksize / step)
                for i in range(0,len(mmp),chunks_per_step):
                    yield mmp[i : i + chunks_per_step][feature]
    
    def iter_id(self,_id,chunksize,step = 1):
        '''
        Iterate over the frames of a single pattern, returning two-tuples of
        (Address,Frames)
        '''
        _id,source,external_id,mmp = self._get_memmap(self.address(_id))
        lmmp = len(mmp)
        for i in xrange(0,lmmp,chunksize):
            data = self._from_skinny(\
                    _id, source, external_id, mmp[i : i + chunksize : step])
            key = i if 1 == step else slice(i,min(i + step,lmmp))
            addr = self._fsfc_address((_id,key))
            yield addr,self.model(data = data)
                                      
        
    
    def update_index(self):
        '''
        Force updates on all indexes. Not sure if this should do anything.
        '''
        self._reindex_if_necessary()
    
    