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
        Return the back-end specific address for a zounds id
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
        Return the total number of frames in the database
        '''
        pass
    
    @abstractmethod
    def list_ids(self):
        '''
        List zounds ids for all existing patterns
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
        Parameters
            _id - a zounds id
        Returns
            source,external_id - the source,external_id pair corresponding to id
        '''
        pass
    
    
    @abstractmethod
    def exists(self,source,external_id):
        '''
        Parameters
            source      - the source of the pattern, e.g. FreeSound, or MySoundDir
            external_id - the identifier assigned to the pattern by the source 
        Returns
            exists - a boolean indicating whether that source,external_id pair
                     exists in the data store
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
        Get row from the database
        
        Parameters
            key - key may be a zounds id, a (source,external_id) pair, or a 
                  backing-store-specific Address instance
        Returns
            rows - If key is an id or a (source,external_id) pair, all rows for
                   the corresponding pattern will be returned. If key is a 
                   backing-store-specific address, all rows specified by the address
                   will be returned
        '''
        pass
    
    @abstractmethod
    def stat(self,feature,aggregate,axis = 0,step = 1):
        '''
        Compute aggregate statistics over all features in the data store
        
        Parameters
            feature   - the feature key or instance for which the statistics will
                        be computed
            aggregate - an aggregate function, e.g., np.sum or np.mean
            axis      - the axis over which the aggregate function will be computed
            step      - the step between collected rows. If the feature's absolute
                        step size is greater than one, redundant data is stored,
                        so it's ok to only sample every n frames.
        Returns
            stat - a numpy array representing the aggregate statistic
        '''
        pass
    
    
    def __getitem__(self,key):
        '''
        Get row from the database
        
        Parameters
            key - key may be a zounds id, a (source,external_id) pair, or a 
                  backing-store-specific Address instance
        Returns
            rows - If key is an id or a (source,external_id) pair, all rows for
                   the corresponding pattern will be returned. If key is a 
                   backing-store-specific address, all rows specified by the address
                   will be returned
        '''
        return self.get(key)
    
    @abstractmethod
    def iter_feature(self,_id,feature,step = 1,chunksize=1):
        '''
        Return an iterator over a single feature from pattern _id
        
        Parameters
            _id       - a zounds id
            feature   - a feature key or instance
            step      - the frequency at which samples should be drawn from frames
            chunksize - if greater than one, chunks of feature values are returned.
                        E.g., if step is 1 and chunksize is 10, chunks of 10 frames
                        will be returned. If step is 2 and chunksize is 10, chunks
                        of 5 frames will be returned.  
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
    def iter_id(self,_id,chunksize,step = 1):
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
        '''
        Allows interchangeable use of Feature instances and their keys
        
        Parameters
            key : A zounds.model.frame.Feature instance, or a Feature key
        Returns
            key, if key is a string, or Feature.key, if key is a Feature.
        '''
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