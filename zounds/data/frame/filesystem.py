from __future__ import division
import os
import shutil
import cPickle
from itertools import repeat
import traceback
import numpy as np
from multiprocessing import Manager,Process
import logging

import zounds.model.frame
from zounds.constants import audio_key,id_key,source_key,external_id_key
from frame import FrameController,UpdateNotCompleteError
from zounds.model.pattern import Pattern
from zounds.util import ensure_path_exists,PoolX,tostring
from zounds.nputil import norm_shape
from zounds.environment import Environment
from time import time

LOGGER = logging.getLogger(__name__)


def update_chunk(chunk_ids,newc_args,env_args,recompute,lock):
    Z = Environment.reanimate(env_args)
    newc = FileSystemFrameController(*newc_args)
    c = Z.framecontroller
    processed = []
    
    for _id in chunk_ids:
        p = Pattern(_id,*c.external_id(_id))
        ec = c.model.extractor_chain(\
                                p,transitional = True, recompute = recompute)
        LOGGER.info('updating %s - %s - %s',_id,p.source,p.external_id)
        newc.append(ec)
        os.remove(c._pattern_path(_id))
        processed.append(_id)
    return processed
    


# TODO: This class really sucks in the case where one, or a few new patterns
# are added to the database.  In that case, it's necessary to go back and pick
# up each small file to recomputed the database-wide stat.
# A couple options
# 1) store all pattern stats in one file. This should make things a lot faster
#
# 2) store only the data necessary. max, min, and sum are easy. mean requires
#    us to store the mean and the number of items used to compute it.  This approach
#    is very efficient, but makes it more difficult to differnentiate between
#    a database update and the addition of a a few patterns. 
class Stats(object):
    
    def __init__(self,controller,datadir):
        object.__init__(self)
        self._path = controller.stat_dn
        self._c = controller
        self._datadir = datadir
    
    def _pattern_stat_filename(self,_id,feature_name,stat_name):
        fn = '%s_%s_%s.dat' % (_id,feature_name,stat_name)
        return os.path.join(self._path,fn)
        
    
    def _db_stat_filename(self,feature_name,stat_name):
        fn = '%s_%s.dat' % (feature_name,stat_name)
        return os.path.join(self._path,fn)
    
    def _get_db_stat(self,feature,op):
        dbsfn = self._db_stat_filename(feature.key,op.__name__)
        if not self._db_feature_file_is_stale(feature.key,op):
            return np.fromfile(dbsfn,dtype = feature.dtype).reshape(feature.dim)
        
        _ids = self._c.list_ids()
        n_ids = len(_ids)
        data = np.ndarray((n_ids,) + norm_shape(feature.dim),dtype = feature.dtype)
        for i,_id in enumerate(_ids):
            data[i] = self._get_pattern_stat(_id, feature,op)
        
        reduced = op(data,axis = 0)
        reduced.tofile(dbsfn)
        return reduced
    
    def _get_pattern_stat(self,_id,feature,op):
        psfn = self._pattern_stat_filename(_id, feature.key, op.__name__)
        
        # TODO: Check if pattern file is fresher than stat file
        if os.path.exists(psfn):
            return np.fromfile(psfn, dtype = feature.dtype).reshape(feature.dim)
        
        feature = self._c[_id][feature.key]
        reduced = op(feature,axis = 0)
        reduced.tofile(psfn)
        return reduced
        
    
    def _db_feature_file_is_stale(self,feature_name,op):
        dbsfn = self._db_stat_filename(feature_name,op.__name__)
        return not os.path.exists(dbsfn) or \
                (os.path.getmtime(dbsfn) < os.path.getmtime(self._datadir))
    
    def _mean_of_squares(self,a):
        return (a**2).mean(0)
    
    def __call__(self,feature,op):
        if op == np.std:
            mean_of_squares = self._get_db_stat(feature, self._mean_of_squares)
            squared_mean = self._get_db_stat(feature,np.mean)**2
            return np.sqrt(mean_of_squares - squared_mean)
            
        
        return self._get_db_stat(feature,op)
    
    


class DataFile(object):
    '''
    A convenience class, for internal use only, that handles the details of
    writing binary meta and feature data to a file.
    '''
    
    def __init__(self,controller):
        '''
        Parameters
            controller - A FrameController-derived instance
        '''
        object.__init__(self)
        self._file = None
        self._id = None
        self._source = None
        self._external_id = None
        self._c = controller
        
    
    def write(self,_id,source,external_id,data):
        '''
        Write data (a numpy.recarray instance) as binary data to a file. Prepend
        metadata (_id, source, and external_id) to the file if this is the first
        call to write().
        '''
        if None is self._file:
            self._id = _id
            self._source = source
            self._external_id = external_id
            self._file = open(self._c._pattern_path(_id),'wb')
            # write the metadata, exlcuding the zounds id, which is stored as
            # the file's name
            source = np.array(source,dtype = 'a%i' % self._c._source_chars)
            external_id = np.array(\
                            external_id,dtype = 'a%i' % self._c._extid_chars)
            self._file.write(source)
            self._file.write(external_id)
        
        # write data ( a numpy record array ), to the file 
        data.tofile(self._file)
        # return the number of frames written to the file
        return len(data)
    
    def close(self):
        self._file.close()

class DummyLock(object):
    def __init__(self):
        object.__init__(self)
    
    def __enter__(self):
        self.acquire()
    
    def __exit__(self,*args):
        self.release()
    
    def acquire(self):
        pass
    
    def release(self):
        pass

# BUG: The locking mechanism here is broken.  It results in a lot of strange
# cPickle errors when unpickling, presumably because multiple processes are
# writing garbled/shuffled data to index.dat, despite my efforts to keep that
# from occurring.
class ConcurrentIndex(object):
    '''
    A convenience class, for internal use only, which handles the details of
    maintaining (hopefully) thread and process safe indexes over the data.  This
    is my first try at implementing "shared" data like this, so I'm sure there
    are some awful bugs.
    '''
    
    def __init__(self,builders,indexpath,datadir,iterator,lock = None):
        '''
        Parameters
            builders - A callable for each index, which takes 
                      (_id,source,external_id,open_file) as parameters, and
                      returns a key and value which will be added to the 
                      corresponding index.
            indexpath - The folder in which the index will be persisted to disk
            datadir   - The folder in which pattern data files live. The modified
                        time on this folder is used to determine whether the 
                        on-disk version of the index has gone stale.
            iterator  - A callable which iterates over all  existing patterns, 
                        returning one set of arguments for the builder functions
                        from each file/pattern.
        '''
        object.__init__(self)
        self._lock = lock or DummyLock()
        self._keys = ['_id','external_id','length']
        self._builders = builders
        # the path to the on-disk index
        self._path = os.path.join(indexpath,'index.dat')
        # the path to a lock file which exists when one instance is re-writing
        # the on-disk index.  It indicates that it's neither safe to write or
        # read.
        self._lock_filename = os.path.join(indexpath,'lock.dat')
        self._datadir = datadir
        self._iterator = iterator
        # the time at which the in-memory index was last refreshed from disk. 
        # Used to determine whether the in-memory version is less current than
        # the on-disk version, as a result of another instance updating it.
        self._in_mem_timestamp = None
        self._data = dict([(k,dict()) for k in self._keys])
        self.startup()
    
    def __getitem__(self,k):
        '''
        Get the most current version of the index with key k
        '''
        if not self.memory_is_stale():
            # The in-memory index is up-to-date
            return self._data[k]
        
        if not self.file_is_stale():
            # The in-memory index was stale, but the file on disk is current.
            # Just load it up and return the value
            with self._lock:
                self._data = self.from_disk()

            return self._data[k]
        
        with self._lock:
            # Both the in-memory and on-disk indexes are stale, rebuild the index,
            # persist it to disk, and return the requested value.
            # rebuild the index
            self.reindex()
            # save it to disk
            self.to_disk()
        
        return self._data[k]
    
    def startup(self):
        '''
        Called by __init__ only. Acquire the freshest index possible, either
        by reading it from disk, or building it your damn self.
        '''
        with self._lock:
            if self.file_is_stale():
                # the on-disk index is stale. Rebuild it for the benefit of self
                # and others
                self.reindex()
                self.to_disk()
            else:
                self._data = self.from_disk()
                
        
    
    def append(self,_id,source,external_id,length):
        '''
        Add a new pattern to the index
        Parameters
            _id         - a zounds id
            source      - the source of the pattern (e.g. FreeSound or MySoundFolder)
            external_id - the identifier assigned to the pattern by source
            length      - the length, in frames, of the pattern 
        '''
        with self._lock:
            if self.memory_is_stale():
                # another controller has updated the disk index. Our in-memory
                # version is out-of-date. Merge the disk version with our version
                # before doing anything else.
                fresh = self.from_disk()
                for k in fresh.iterkeys():
                    self._data[k].update(fresh[k])
            
            # Add the new pattern to all relevant indexes and persist them to disk
            self._data['_id'][_id] = (source,external_id)
            self._data['external_id'][(source,external_id)] = _id
            self._data['length'][_id] = length
            self.to_disk()
            self._in_mem_timestamp = time()
        
    
    
    def from_disk(self):
        '''
        Read the index from disk
        '''
        with open(self._path,'rb') as f:
            index = cPickle.load(f)
        self._in_mem_timestamp = time()
        return index
    
    def to_disk(self):
        '''
        Persist the in-memory index to disk
        ''' 
        mode = 'wb' if not os.path.exists(self._path) else 'rw+b'
        with open(self._path,mode) as f:
            cPickle.dump(self._data,f,cPickle.HIGHEST_PROTOCOL)
        self._in_mem_timestamp = time()
        
    
    def reindex(self):
        '''
        Iterate over all existing patterns, recording any metadata required
        by the index
        '''
        for metadata in self._iterator(self._data['_id']):
            for index_key in self._keys:
                builder = self._builders[index_key]
                k,v = builder(*metadata)
                self._data[index_key][k] = v
    
    def force(self):
        with self._lock:
            self.reindex()
            self.to_disk()
        

    def id_from_external_id(self,extid):
        '''
        Paramters
            extid - a two tuple of (source,external_id)
        Returns
            the zounds id that corresponds to the (source,external_id) pair
        '''
        return self._data['external_id'][extid]
    
    def external_id_from_id(self,_id):
        '''
        Parameters
            _id - a zounds id
        Returns
            the (source,external_id) pair corresponding to the zounds id
        '''
        return self._data['_id'][_id]
    
    def pattern_length(self,_id):
        '''
        Paramters
            _id - a zounds id
        Returns
            the length of the pattern with zounds id, in frames
        '''
        return self._data['length'][_id]
    
    
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
    '''
    A backing store that persists patterns as a simple collection of binary
    files, and mantains necessary indexes over that data as pickled python
    dictionaries.
    
    The structure of the directory is as follows:
    
    /path/to/data
        data
            35b412ad41f74c1f9d56f2ae6f757393.dat
            4331530721d9454b89db30713801b053.dat
        feature.dat
        index.dat
    
    Where
        - data is a directory containing binary feature data for each pattern
        - feature.dat is a file containing the pickled set of current features
        - index.dat contains pickled dictionaries which serve as indexes into the
          data. 
              - _ids - maps zounds ids -> (source,external_id) pairs
              - _external_ids - maps (source,external_id) pairs to zounds ids
              - lengths - maps zounds ids to their lenghts in frames
        
    '''
    
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
        
        # BUG: The following two methods only work/make sense if the key is a
        # slice.    
        @property
        def start(self):
            return self._index.start
        
        @property
        def stop(self):
            return self._index.stop
        
        
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
    stat_dn = 'stat'
    
    def __init__(self,framesmodel,filepath,lock = None):
        '''
        Parameters
            framesmodel - a zounds.model.frame.Frames-derived class that defines
                          the features that should be computed and stored for
                          each pattern
            filepath    - a path to the directory that should contain all data 
                          files. It's ok if this directory doesn't yet exist.
        '''
        FrameController.__init__(self,framesmodel)
        self.lock = lock
        # The number of bytes to reserve at the beginning of each file for the
        # source's name
        self._source_chars = 32
        # The number of bytes to reserve at the begninning of each file for the
        # external_id
        self._extid_chars = 32
        # The total number of bytes needed at the beginning of each file for
        # metadata
        self._metadata_chars = self._source_chars + self._extid_chars
        self._load(filepath)
        
    
    def _load(self,filepath):
        self._rootdir = filepath
        
        # build the path to the directory which will house pattern files
        self._datadir = \
            os.path.join(self._rootdir,FileSystemFrameController.data_dn)
        # build the path on disk
        ensure_path_exists(self._datadir)
        
        ensure_path_exists(FileSystemFrameController.stat_dn)
        
        fsfc = FileSystemFrameController
        
        self._stats = Stats(self,self._datadir)
        
        # build the path to the features file
        self._feature_path = os.path.join(self._rootdir,fsfc.features_fn)
        # build the path to a special file whose existence indicates that an
        # update is in progress
        self._sync_path = os.path.join(self._rootdir,fsfc.sync_fn)
        # load the features from disk
        self._load_features()
        
        # Features that are redundant (i.e., the same for every frame), and
        # won't be stored on disk as part of the recarray
        self._excluded_metadata = [id_key,source_key,external_id_key]
        
        # sort the tuples of (key,(shape,dtype,step)), so that they're always
        # in the same order
        dims = sorted(self._dimensions.items())
        self._dtype = []
        self._skinny_features = []
        for k,v in dims:
            shape,dtype,step = v
            # append everything th _dtype. This is the data type of the fully
            # fleshed out record array, i.e., the one that contains columns
            # for _id, source, and external_id
            self._dtype.append((k,dtype,shape))
            if k not in self._excluded_metadata:
                # leave out features that would be redundant to store for each
                # frame of audio
                self._skinny_features.append((k,dtype,shape))
        
        
        self._np_dtype = np.dtype(self._dtype)
        self._skinny_dtype = np.dtype(self._skinny_features)
        self._reindex_if_necessary()
    
    
    def __repr__(self):
        return tostring(self,model = self.model,data_dir = self._rootdir)
    
    def _reindex_if_necessary(self):
        # create an index instance, which will get us the most up-to-date
        # information, in the most efficient way possible
        builders = {'_id' : self._build_id_index,
                    'external_id' : self._build_external_id_index,
                    'length' : self._build_lengths_index}
        self._index = ConcurrentIndex(\
                    builders,self._rootdir,self._datadir,
                    self._iter_patterns,lock = self.lock)
    
    
    def _to_skinny(self,record):
        '''
        Return only the fields of the record that will be stored on disk
        '''
        return record[self._skinny_features]
    
    
    def _from_skinny(self,_id,source,external_id,skinny_record):
        '''
        "Hydrate" the data, by adding columns for metadata, namely, _id, source,
        and external_id.
        '''
        r = np.recarray(skinny_record.shape,self._np_dtype)
        r[id_key] = _id
        r[source_key] = source
        r[external_id_key] = external_id
        for sf in self._skinny_features:
            name,dtype,shape = sf
            r[name] = skinny_record[name]
        return r

    def _recarray(self,rootkey,data,done = False):
        '''
        Build a "skinny" record array, i.e., one that excludes metadata columns
        '''
        meta,record = FrameController._recarray(\
                self, rootkey, data, done = done, 
                dtype = self._skinny_dtype, meta = self._excluded_metadata)
        _id,source,external_id = meta
        return _id,source,external_id,record
    
    def _str_dtype(self,nchars):
        return 'a%i' % nchars
    
    def _get_memmap(self,addr):
        '''
        Get a memory mapped view into a pattern file, accounting for leading
        bytes dedicated to metadata, and any frame offset indicated by addr.
        '''
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
        '''
        Parameters
            f - an open file instance, containing pattern data
        Returns
            source,extid - the metadata from the beginning of the file
        '''
        meta = f.read(self._metadata_chars)
        source = np.fromstring(meta[:self._source_chars],
                                   dtype = self._str_dtype(self._source_chars))[0]
        extid = np.fromstring(meta[self._source_chars:],
                                  dtype = self._str_dtype(self._extid_chars))[0]
        return source,extid
    
    def _get_hydrated(self,addr):
        '''
        Paramters
            addr - A FileSystemFrameController.Address instance
        Returns
            rec - A fully "hydrated" recarray instance, which contains columns
                  for metadata
        '''
        return self._from_skinny(*self._get_memmap(addr))
        
    
    def _pattern_path(self,_id):
        '''
        Parameters
            _id - a zounds id
        Returns
            path - the path to the binary file containing feature data for the
                   pattern with id
        '''
        return os.path.join(self._datadir,
                            _id + FileSystemFrameController.file_extension)
    
    
    def _load_features(self):
        '''
        Ensure that the features of the current FrameModel are stored on disk
        '''
        try:
            # The current feature set has already been persisted to disk. Simply
            # load it in to memory
            with open(self._feature_path,'r') as f:
                self._features,self._dimensions = cPickle.load(f)
        except IOError:
            # The current feature set must be persisted to disk
            self._features = self.model.features
            self._dimensions = self.model.dimensions()
            with open(self._feature_path,'w') as f:
                cPickle.dump((self._features,self._dimensions),f)        
        
        
    def _build_id_index(self,_id,source,external_id,f):
        '''
        A function, to be passed to a ConcurrentIndex instance as part of the
        builders parameter, which maps zounds ids to (source,external_id) pairs.
        '''
        return _id,(source,external_id)
    
    def _build_external_id_index(self,_id,source,external_id,f):
        '''
        A function, to be passed to a ConcurrentIndex instance as part of the
        builders parameter, which maps (source,external_id) pairs to zounds ids.
        '''
        return (source,external_id),_id    
    
    def _build_lengths_index(self,_id,source,external_id,f):
        '''
        A function, to be passed to a ConcurrentIndex instance as part of the
        builders parameter, which maps zounds ids to their respective lenght
        in frames
        '''
        fs = os.path.getsize(os.path.abspath(f.name))
        nframes = (fs - self._metadata_chars) / self._skinny_dtype.itemsize
        return _id,nframes
        
    def _iter_patterns(self,d):
        '''
        A function, to be passed to a ConcurrentIndex instance as the iterator
        parameter, which iterates over all existing patterns.
        '''
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
        '''
        A convenience property to access the id -> (source,external_id) index
        '''
        return self._index['_id']
    
    @property
    def _external_ids(self):
        '''
        A convenience property to access the (source,external_id) -> id index
        '''
        return self._index['external_id']
    
    @property
    def _lengths(self):
        '''
        A convenience property to access the id -> length-in-frames index
        '''
        return self._index['length']
    
    def __len__(self):
        return np.sum(self._lengths.values())
    
    def list_ids(self):
        return set(self._ids.keys())
    
    def list_external_ids(self):
        return self._external_ids.keys()
    
    def external_id(self,_id):
        return self._ids[_id]
    
    def exists(self,source,external_id):
        return (source,external_id) in self._external_ids
    
    @property
    def _temp_path(self):
        '''
        For use during a sync. Return a modified version of the current path.
        '''
        return '%s_sync' % self._rootdir
    
    def sync(self,add,update,delete,recompute):
        if Environment.parallel() and \
           self.concurrent_reads_ok and \
           self.concurrent_writes_ok: 
            self._sync_multi(add, update, delete, recompute)
        else:
            self._sync_single(add, update, delete, recompute)
    
    # TODO: A lot of this code is identical to what's in PyTablesFrameController.sync()
    def _sync_single(self,add,update,delete,recompute):
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
            
            LOGGER.info('updating %s - %s',p.source,p.external_id)
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
    
    def _sync_multi(self,add,update,delete,recompute):
        newc = FileSystemFrameController(self.model,self._temp_path)
        new_ids = newc.list_ids()
        _ids = self.list_ids()
        difference = list(_ids - new_ids)
        mgr = Manager()
        lock = mgr.Lock()
        
        chunksize = 15
        newc_args = (self.model,self._temp_path,lock)
        # make sure that Environment instances created in new processes don't
        # initiate their own sync operations, and that controllers created in 
        # new processes have a reference to the multiprocess lock
        env_args = self.model.env().__getstate__(lock = lock, do_sync = False)
        args = []
        for i in range(0,len(difference),chunksize):
            chunk_ids = difference[i : i + chunksize]
            args.append((chunk_ids,newc_args,env_args,recompute,lock))
        

        PoolX(Environment.n_cores).map(update_chunk,args)
        
        self._index.force()
        newc._index.force()
        
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
        _id = datafile._id
        source = datafile._source
        external_id = datafile._external_id
        # KLUDGE: I'm still experiencing some transient cPickle errors from
        # ConcurrentIndex. They're nothing serious; the worst thing that could
        # happen is a duplicate or two getting inserted into the db because
        # two processes didn't have external_id indexes in sync.  For now, I'm
        # going to yell about it, but nothing else.
        try:
            self._index.append(_id, source, external_id, nframes)
        except:
            LOGGER.exception(traceback.format_exc())
        addr = self.address_class((_id,slice(0,nframes)))
        datafile.close()
        return addr
    
    def address(self,_id):
        return self.address_class((_id,slice(0,self._lengths[_id])))
    
    def get(self,key):
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
        if axis > 0:
            # for now, I'm only implementing feature-wise stats
            raise NotImplemented
        
        if step > 1:
            # for now, I'm only implementing step sizes of one
            raise NotImplemented
        
        return self._stats(feature,aggregate)
        
    
    def get_features(self):
        return self._features
    
    def get_dtype(self,key):
        key = self._feature_as_string(key)
        dtype = self._np_dtype[key]
        try:
            return dtype.subdtype[0]
        except TypeError:
            return dtype
    
    def get_dim(self,key):
        key = self._feature_as_string(key)
        dtype = self._np_dtype[key]
        try:
            return dtype.subdtype[1]
        except TypeError:
            return ()
    
    def iter_all(self,step = 1):
        _ids = self.list_ids()
        for _id in _ids:
            _id,source,external_id,mmp = self._get_memmap(self.address(_id))
            lmmp = len(mmp)
            for i in xrange(0,lmmp,step):
                key = i if step == 1 else slice(i,min(i + step,lmmp))
                addr = self.address_class((_id,key))
                data = self._from_skinny(_id, source, external_id, mmp[key])
                yield addr,self.model(data = data)

    def iter_feature(self,_id,feature,step = 1, chunksize = 1):
        # for metadata, return an iterator that outputs the correct value as
        # many times as necessary
        
        # for non-metdata, iterate over a memmapped file.
        feature = self._feature_as_string(feature)
        l = self._lengths[_id]
        _id,source,external_id,mmp = self._get_memmap(\
                self.address_class((_id,slice(0,l,step))))
        
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
        _id,source,external_id,mmp = self._get_memmap(self.address(_id))
        lmmp = len(mmp)
        for i in xrange(0,lmmp,chunksize):
            data = self._from_skinny(\
                    _id, source, external_id, mmp[i : i + chunksize : step])
            key = i if 1 == step else slice(i,min(i + step,lmmp))
            addr = self.address_class((_id,key))
            yield addr,self.model(data = data)
                                      
        
    
    def update_index(self):
        self._reindex_if_necessary()
    

# Crazy Bad KLUDGE: I rely on FrameController-derived classes to define a back-end
# specific Address class as a nested class. This makes those classes impossible
# to pickle, however.  This is a baaad solution, but I'd like to keep moving
# for right now.    
import sys         
setattr(sys.modules[__name__], 'Address', FileSystemFrameController.Address)