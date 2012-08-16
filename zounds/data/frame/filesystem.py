from __future__ import division
import os
import cPickle
import struct
import numpy as np
from frame import FrameController
from zounds.util import ensure_path_exists

'''
Zounds File Format
MetaData
8*32=256 - source
8*32=256 - external_id

Total offset = 256 + 256 = 512 bits = 68 bytes

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
    
    file_extension = '.dat'
    features_fn = 'features' + file_extension
    external_ids_fn = 'external_ids' + file_extension
    sync_fn = 'sync' + file_extension
    lengths_fn = 'lengths' + file_extension
    data_dn = 'data'
    metadata_fmt = '64c'
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self.framesmodel)
        self._rootdir = filepath
        self._datadir = \
            os.path.join(self._rootdir,FileSystemFrameController.data_dn)
        ensure_path_exists(self._datadir)
        self._feature_path = \
            os.path.join(self._rootdir,FileSystemFrameController.features_fn)
        self._external_ids_path = \
            os.path.join(self._rootdir,FileSystemFrameController.external_ids_fn)
        self._lengths_path = \
            os.path.join(self._rootdir,FileSystemFrameController.lengths_fn)
        self._sync_path = \
            os.path.join(self._rootdir,FileSystemFrameController.sync_fn)
        self._dtype = [(k,v[1],v[0]) for k,v in \
                        self.model.dimensions().iteritems()]
        self._np_dtype = np.dtype(self._dtype)
        self._store_features()
        self._external_ids = self._load_external_id_index()
        self._lengths = self._load_lengths_index()
    
    @property
    def record_size(self):
        return self._np_dtype.itemsize
    
    def _file_is_stale(self,path):
        '''
        Compare an index file with the modified time of the data directory.  
        Return True if the data directory has been modified more recently than
        the index file.
        '''
        return os.path.getmtime(path) < os.path.getmtime()
    
    def _store_features(self):
        if os.path.exists(self._feature_path):
            return
        with open(self._feature_path) as f:
            cPickle.dump(self.model.features,f)
    
    # TODO: Refactor the following four methods to loop over data files once
    def _load_external_id_index(self):
        eip = self._external_ids_path
        if os.path.exists(eip) and not self._file_is_stale(eip):
            with open(eip) as f:
                return cPickle.load(f)
            
        index = self._build_external_id_index()
        with open(eip) as f:
            cPickle.dump(index,f)
        return index
    
    def _build_external_id_index(self):
        index = dict()
        files = os.listdir(self._datadir)
        for f in files:
            _id = f.split('.')[0]
            path = os.path.join(self._datadir,f)
            with open(path) as f:
                metadata = struct.unpack(\
                        FileSystemFrameController.metadata_fmt,f.read(64))
                source = metadata[:32]
                external_id = metadata[32:]
                index[(source,external_id)] = _id
        return index
    
    def _load_lengths_index(self):
        lp = self._lengths_path
        if os.path.exists(lp) and not self._file_is_stale(lp):
            with open(lp) as f:
                return cPickle.load(f)
        
        index = self._build_lengths_index()
        with open(lp) as f:
            cPickle.dump(index,f)
        return index
        
    def _build_lengths_index(self):
        index = dict()
        files = os.listdir(self._datadir)
        for f in files:
            _id = f.split('.')[0]
            path = os.path.join(self._datadir,f)
            fs = os.path.getsize(path)
            # ignore the metadata when computing the file size
            nframes = (fs - 64) / self.record_size
            index[_id] = nframes
        return index
            
        
    
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
        raise NotImplemented()
    
    def exists(self,source,external_id):
        '''
        Return true if the source,external_id pair is in the in-memory 
        external_id -> _id mapping
        '''
        raise NotImplemented()
    
    def sync(self,add,update,delete,chain):
        '''
        When need for a sync is detected, a sync file is created.  Keep working until
        no files exist with creation dates prior to the sync file's creation date
        '''
        raise NotImplemented()
    
    def append(self,extractor_chain):
        '''
        Create a new file. Add meta and real data
        '''
        raise NotImplemented()
    
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
    
    