from __future__ import division
import os
import shutil
import cPickle
import struct
from itertools import repeat
import numpy as np

import zounds.model.frame
from zounds.constants import audio_key,id_key,source_key,external_id_key
from frame import FrameController,UpdateNotCompleteError
from zounds.model.pattern import Pattern
from zounds.util import ensure_path_exists

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

class DataFile(object):
    
    def __init__(self):
        self._file = None
        self._id = None
        self._source = None
        self._external_id = None
        
    
    def write(self,_id,source,external_id,data):
        if None is self._file:
            self._id = _id
            self._source = source
            self._external_id = external_id
            fsfc = FileSystemFrameController
            self._file = open(fsfc._pattern_path(_id),'wb')
            source = np.array(source,dtype = 'a%i' % fsfc._source_chars)
            external_id = np.array(source,dtype = 'a%i' % fsfc._extid_chars)
            self._file.write(source)
            self._file.write(external_id)
        
        self._file.write(data.tostring())
        return len(data)
    
    def close(self):
        self._file.close()
        
        
            
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
                self._len = len(self._index)
                self._index = np.array(self._index)
                self.min = self._index.min()
                self.max = self._index.max()
            except TypeError:
                pass
            
            try:
                self._len = self._index.stop - self._index.start
                if key.step not in [None,1]:
                    raise ValueError(\
                    'When using a slice as an address key, it must have a step of 1')
                self.min = self._index.start
                self.max = self._index.stop - 1
            except AttributeError:
                pass
            
            if isinstance(self._index,int):
                self._len = 1
                self.min = self._index
                self.max = self._index
            else:
                raise ValueError(\
                    'key must be a two-tuple of (_id,int, iterable of ints, or slice)')
            
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
                raise ValueError('Comparisons are meaningless for addresses with different ids')
        
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
    external_ids_fn = 'external_ids' + file_extension
    ids_fn = '_ids' + file_extension
    sync_fn = 'sync' + file_extension
    lengths_fn = 'lengths' + file_extension
    data_dn = 'data'
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self.framesmodel)
        self._source_chars = 32
        self._extid_chars = 32
        self._metadata_chars = self._source_chars + self._extid_chars
        self._metadata_fmt = '%ic' % self._metadata_chars
        self._load(filepath)
        
    
    def _load(self,filepath):
        self._rootdir = filepath
        
        # TODO: Refactor this!
        self._datadir = \
            os.path.join(self._rootdir,FileSystemFrameController.data_dn)
        ensure_path_exists(self._datadir)
        self._feature_path = \
            os.path.join(self._rootdir,FileSystemFrameController.features_fn)
        self._external_ids_path = \
            os.path.join(self._rootdir,FileSystemFrameController.external_ids_fn)
        self._ids_path = \
            os.path.join(self._rootdir,FileSystemFrameController.ids_fn)
        self._lengths_path = \
            os.path.join(self._rootdir,FileSystemFrameController.lengths_fn)
        self._sync_path = \
            os.path.join(self._rootdir,FileSystemFrameController.sync_fn)
        
        # Get the keys and shapes of all stored features
        dims = self.model.dimensions
        self._dtype = [(k,v[1],v[0]) for k,v in dims.iteritems()]
        self._np_dtype = np.dtype(self._dtype)
        self._store_features()
        
        # TODO: Perhaps these should be loaded lazily
        self._external_ids,self._lengths,self._ids = self._load_indexes(*[
                                                               
                    (self._build_external_id_index, self._external_ids_path),
                    (self._build_lengths_index,     self._lengths_path),
                    (self._build_ids_index,         self._ids_path)
        ])
    
        # Features that are redundant (i.e., the same for every frame), and
        # won't be stored on disk as part of the recarray
        self._excluded_metadata = [id_key,source_key,external_id_key]
        # Only the feature keys that will be stored, and aren't redundant metadata
        self._skinny_features = filter(\
            lambda a : a not in self._excluded_metadata,dims.iterkeys())
        # The dtype of the recarrays that will be stored on disk
        self._skinny_dtype = np.dtype(filter(\
            lambda a : a[0] in self._skinny_features,
            self._np_dtype.fields.iteritems()))
    
    
    def _to_skinny(self,record):
        # return only the fields of the record that will be stored on disk
        return record[self._skinny_features]
    
    def _from_skinny(self,_id,source,external_id,skinny_record):
        r = np.recarray(len(skinny_record),self._np_dtype)
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
            meta = f.read(self._metadata_chars)
            source = np.fromstring(meta[:self._source_chars])
            extid = np.fromstring(meta[self._source_chars:])
            # get the offset, in bytes, where we'll start reading
            start_bytes = start_index * size 
            data = np.memmap(\
                    f,dtype = self._skinny_dtype, mode = 'r', 
                    offset = start_bytes + self._metadata_chars)
        return _id,source,extid,data[:to_read:step]
    
    def _get_hydrated(self,addr):
        return self._from_skinny(*self._get_memmap(addr))
        
    
    def _pattern_path(self,_id):
        return os.path.join(self._datadir,
                            _id + FileSystemFrameController.file_extension)
    
    @property
    def record_size(self):
        return self._np_dtype.itemsize
    
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
        self._features = self.model.features
        
        if os.path.exists(self._feature_path):
            return
        
        with open(self._feature_path,'w') as f:
            cPickle.dump(self._features,f)
        
        
    
    def _load_indexes(self,*paths):
        '''
        Load indexes into memory by either reading them from disk, if they're
        current, or rebuilding them.  If an index is rebuilt, the old index
        will be overwritten by the new one.
         
        paths should be a list of tuples of (function_to_build_index,index_file_name)
        '''
        lp = len(paths)
        indexes = [None] * lp
        builders = []
        
        for i,p in enumerate(paths):
            builder,path = p
            if os.path.exists(path) and not self._file_is_stale(path):
                # The index is already built, and is current. Just read it from
                # disk.
                with open(path,'r') as f:
                    indexes[i] = cPickle.load(f)
            else:
                # Mark this index to be rebuilt
                builders.append((i,builder,path))
        
        if builders:
            # Some indexes need to be rebuilt. Pass them along to a function
            # which will iterate over all data files once, passing necessary
            # data to each index builder.
            built = self._iter_patterns(*builders)
            for i,index in built:
                # assign the newly built index to the correct position in the
                # output list
                indexes[i] = index
                builder,path = paths[i]
                # write the newly built index to disk
                with open(path,'w') as f:
                    cPickle.dump(index,f)
    
        return indexes
    
    def _build_id_index(self,f,_id,source,external_id,path,index):
        index[_id] = (source,external_id)
    
    def _build_external_id_index(self,f,_id,source,external_id,path,index):    
        index[(source,external_id)] = _id
    
    def _build_lengths_index(self,f,_id,source,external_id,path,index):
        fs = os.path.getsize(path)
        nframes = (fs - self._metadata_chars) / self.record_size
        index[_id] = nframes
        
    def _iter_patterns(self,*index_builders):
        files = os.listdir(self._datadir)
        for f in files:
            _id = f.split('.')[0]
            path = os.path.join(self._datadir,f)
            for i,func,index in index_builders:
                with open(path,'r') as of:
                    metadata = struct.unpack(\
                            self._metadata_fmt,of.read(self._metadata_chars))
                    source = metadata[:self._source_chars]
                    external_id = metadata[self._source_chars:]
                    func(f,_id,source,external_id,path,index)
        return [(i,index) for i,index in index_builders]
    
    @property
    def concurrent_reads_ok(self):
        return True
    
    @property
    def concurrent_writes_ok(self):
        return True
    
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
        return set(self._lengths.keys())
    
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
        data = dict([(k,None) for k in self.steps.iterkeys()])
        chunks_processed = 0
        nframes = 0
        datafile = DataFile()
        for k,v in chain.process():
            if k not in self.steps:
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
        
        _id = datafile._id
        ext_id = (datafile._source,datafile._external_id)
        datafile.close()
        # update indexes
        self._ids[ext_id] = _id
        self._external_ids[_id] = ext_id
        self._lengths[_id] = nframes
        return FileSystemFrameController.Address((_id,slice(None)))
    
    def address(self,_id):
        '''
        Return the address for an _id
        '''
        return FileSystemFrameController.Address((_id,slice(None)))
    
    def get(self,key):
        '''
        Zounds ID - Fetch all the data from the file with name ID
        
        (Source,External Id) Lookup the Zounds ID from the in-memory hashtable.
          Load the file.
          
        Address (Zounds ID,Offset,Length) - Open a memmapped version of the file,
        Read only the frames specified
        '''
        
        _id = None
        addrcls = FileSystemFrameController.Address
        if isinstance(key,str):
            # the key is a zounds id
            addr = addrcls((key,slice(None)))
        elif isinstance(key,tuple) \
            and 2 == len(key) \
            and all([isinstance(k,str) for k in key]):
            # the key is a (source, external id) pair
            _id = self._external_ids[key]
            addr = addrcls((_id,slice(None)))
        elif isinstance(key,addrcls):
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
    
    def iter_feature(self,_id,feature,step = 1, chunksize = 1):
        '''
        For file with Zounds ID, iterate over the feature in the manner specified.
        TODO: What does the underlying implementation of memmap[::step] do?
        '''
        # for metadata, return an iterator that outputs the correct value as
        # many times as necessary
        
        # for non-metdata, iterate over a memmapped file.
        feature = feature if isinstance(feature,str) else feature.key
        _id,source,external_id,mmp = self._get_memmap(\
                FileSystemFrameController.Address((_id,slice(None,None,step))))
        
        meta = feature in self._excluded_metadata
        if meta:
            return repeat(locals[feature],len(mmp))
        else:
            if 1 == chunksize:
                for row in mmp:
                    yield row[feature]
            else:
                chunks_per_step = int(chunksize / step)
                for i in range(0,len(mmp),chunks_per_step):
                    yield mmp[i : i + chunks_per_step]
        
    
    def get_features(self):
        '''
        Return a dictionary mapping Feature Key -> Feature for the current FrameModel
        '''
        with open(self._feature_path,'r') as f:
            return cPickle.load(f)
    
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
        raise NotImplemented()
    
    def iter_id(self,step = 1):
        '''
        Iterate over the frames of a single pattern, returning two-tuples of
        (Address,Frames)
        '''
        raise NotImplemented()
    
    def update_index(self):
        '''
        Force updates on all indexes. Not sure if this should do anything.
        '''
        raise NotImplemented()
    
    