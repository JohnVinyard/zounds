from abc import ABCMeta,abstractmethod,abstractproperty
import numpy as np
import zounds
from zounds.data.controller import Controller
from zounds.nputil import pad

class FrameController(Controller):
    __metaclass__ = ABCMeta
    
    def __init__(self,framesmodel):
        Controller.__init__(self)
        self.model = framesmodel
    
    @abstractmethod
    def address(self,_id):
        '''
        Return the address for a specific _id
        '''
        pass
    
    @abstractproperty
    def concurrent_reads_ok(self):
        '''
        Return True if it's safe for multiple processes to read from the DB at
        once
        '''
        pass
    
    @abstractproperty
    def concurrent_writes_ok(self):
        '''
        Return True if it's safe for multiple processes to write to the DB at 
        once
        '''
        pass
        
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
    def list_external_ids(self):
        '''
        List all pattern (source,external_id) pairs
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
    
    @abstractmethod
    def stat(self,feature,aggregate,axis = 0,step = 1):
        pass
    
    
    def __getitem__(self,key):
        return self.get(key)
    
    @abstractmethod
    def iter_feature(self,_id,feature,step = 1,chunksize=1):
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
    
    @abstractmethod
    def iter_id(self,step = 1):
        '''
        Iterate over the frames from a single id, returning two-tuples of 
        (address,frames)
        '''
        pass
    
    @abstractmethod
    def update_index(self):
        '''
        Force updates on all table indexes. This may do nothing, depending
        on the backing store.
        '''
        pass
    
    @property
    def address_class(self):
        return self.__class__.Address
        
    def _feature_as_string(self,key):
        if isinstance(key,zounds.model.frame.Feature):
            return key.key
        
        if isinstance(key,str):
            return key
        
        raise ValueError(\
            'key must be a zounds.model.frame.Feature instance, or a string')
            
    
    def _recarray(self,rootkey,data,done = False,dtype = None,meta = None):
        if done:
            # This is the last chunk. Write as many frames as the root feature
            # has left.
            l = len(data[rootkey])
        else:
            # This is not the last chunk. Write as many frames as the feature
            # with the fewest frames computed.
            srt = sorted([len(a) for a in data.itervalues()])
            l = srt[0]
        
        dtype = np.dtype(self.recarray_dtype) if dtype is None else dtype
        record = np.recarray(l,dtype)
        
        for k in dtype.names:
            # Assign the feature data to the recarray, ensuring that it's long
            # enough by padding with zeros.
            record[k] = pad(data[k][:l],l)
            # Chop off the data we've just written
            data[k] = data[k][l:]
        
        if None is meta:
            return record
        
        return [data[k][0] for k in meta],record

class UpdateNotCompleteError(BaseException):
    '''
    Raised when a db update fails
    '''
    def __init__(self):
        BaseException.__init__(self,Exception('The PyTables update failed'))