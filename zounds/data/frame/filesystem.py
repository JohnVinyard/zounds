from __future__ import division
import os
import shutil
import cPickle
import struct
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
        self._dtype = [(k,v[1],v[0]) for k,v in \
                        self.model.dimensions().iteritems()]
        self._np_dtype = np.dtype(self._dtype)
        self._store_features()
        
        # TODO: Perhaps these should be loaded lazily
        self._external_ids,self._lengths,self._ids = self._load_indexes(*[
                                                               
                    (self._build_external_id_index, self._external_ids_path),
                    (self._build_lengths_index,     self._lengths_path),
                    (self._build_ids_index,         self._ids_path)
        ])
    
        
    
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
    
    def list_ids(self):
        '''
        Read files from the directory where data is stored. Keep it in memory. Update
        the in-memory list as append() is called
        '''
        return set(self._lengths.keys())
    
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
    
    def external_id(self,_id):
        '''
        Read the external_id from the file's metadata and cache it.
        '''
        return self._ids[_id]
    
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
        for k,v in chain.process():
            if k not in self.steps:
                continue
    
            if None is abs_steps[k]:
                # The first call to _process has been made, so it's safe to
                # get the absolute step value for this extractor
                abs_steps[k] = chain[k].step_abs()
            
            if rootkey == k and chunks_processed > 0:
                record = self._recarray(rootkey, data, done = False)
                # TODO: Write differently
                self._ensure_lock_and_append(record)
                nframes += len(record)
                
            
            f = np.repeat(v, abs_steps[k], 0)
            data[k] = safe_concat(data[k],f)
            if rootkey == k:
                chunks_processed += 1
        
        
        record = self._recarray(rootkey, data, done = True)
        # TODO: Write differently
        self._ensure_lock_and_append(record)
        nframes += len(record)
    
        # TODO: Move these hardcoded values (_id,source,external_id) somewhere more central
        row = record[0]
        _id = row[id_key]
        source = row[source_key]
        external_id = row[external_id_key]
        # update indexes
        self._ids[(source,external_id)] = _id
        self._external_ids[_id] = (source,external_id)
        self._lengths[_id] = nframes
        return FileSystemFrameController.Address((_id,slice(None)))
    
    def get(self,key):
        '''
        Zounds ID - Fetch all the data from the file with name ID
        
        (Source,External Id) Lookup the Zounds ID from the in-memory hashtable.
          Load the file.
          
        Address (Zounds ID,Offset,Length) - Open a memmapped version of the file,
        Read only the frames specified
        '''
        raise NotImplemented()
    
    def stat(self,feature,aggregate,axis = 0,step = 1):
        '''
        Streaming, multiprocess min,max,sum,mean,std implementations.
        
        Store the results for each file in a companion file.
        
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
        raise NotImplemented()
    
    def get_features(self):
        '''
        Return a dictionary mapping Feature Key -> Feature for the current FrameModel
        '''
        raise NotImplemented()
    
    def get_dtype(self,key):
        '''
        Return the numpy datatype of the feature with key
        '''
        raise NotImplemented()
    
    def get_dim(self,key):
        '''
        Return the dimension of the feature with key
        '''
        raise NotImplemented()
    
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
    
    