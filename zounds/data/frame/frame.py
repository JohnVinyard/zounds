from abc import ABCMeta,abstractmethod,abstractproperty
import numpy as np
import zounds
from zounds.data.controller import Controller
from zounds.nputil import pad
from zounds.util import tostring

class FrameController(Controller):
    '''
    Handle persistence and retrieval of 
    :py:class:`~zounds.model.frame.Frames`-derived class instances.
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self,framesmodel):
        Controller.__init__(self)
        self.model = framesmodel
    
    def __repr__(self):
        return tostring(self,model = self.model)
    
    def __str__(self):
        return self.__repr__()
    
    @abstractmethod
    def address(self,_id):
        '''
        Return the back-end specific address for a zounds id
        '''
        pass
    
    @abstractproperty
    def concurrent_reads_ok(self):
        '''
        Return :code:`True` if it's safe for multiple processes to read from the DB at
        once
        '''
        pass
    
    @abstractproperty
    def concurrent_writes_ok(self):
        '''
        Return :code:`True` if it's safe for multiple processes to write to the DB at 
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
        List (source,external_id) pairs for all existing patterns
        '''
        pass
    
    @abstractmethod
    def external_id(self,_id):
        '''
        Get the external id belonging to zounds id
        
        :param _id: a zounds id
        
        :returns: A two-tuple of (source,external_id). The source,external_id \
        pair corresponding to the zounds id
        '''
        pass
    
    
    @abstractmethod
    def exists(self,source,external_id):
        '''
        :param source: the source of the pattern, e.g. FreeSound, or MySoundDir
        
        :param external_id: the identifier assigned to the pattern by the source
         
        :returns: a boolean indicating whether that source,external_id pair \
        exists in the data store
        '''
        pass
     
    
    @abstractmethod
    def sync(self,add,update,delete,chain):
        '''
        This method is called when the user defined set of 
        :py:class:`~zounds.model.frame.Feature`-derived instances belonging to 
        the :py:class:`~zounds.model.frame.Frames`-derived class for this 
        application has changed.
        
        The controller is informed about which features will added, updated, 
        and deleted, and should update stored data in a manner appropriate to
        the backing store.
        
        Note that this might be a long running process, so implementations 
        should be able to save state and resume in case of an error or 
        interruption.
        
        :param add: A list of features that are new
        
        :param update: A list of features that should be recomputed
        
        :param delete: A list of features that have been removed
        
        :param chain: An :py:class:`~zounds.analyze.extractor.ExtractorChain` \
        instance which has been built to compute the new feature graph in the \
        most efficient way possible.
        
        '''
        pass
    
    @abstractmethod
    def append(self,extractor_chain):
        '''
        Add a single new pattern to the data store.
        
        :param extractor_chain: An \
        :py:class:`~zounds.analyze.extractor.ExtractorChain` instance which can \
        compute the current feature graph.
        '''
        pass
    
    @abstractmethod
    def get(self,key):
        '''
        Get rows from the database
        
        :param key: :code:`key` may be
        
            * a zounds id
            * a two-tuple of (source,external_id)
            * a backing-store-specific :py:class:`zounds.model.frame.Address`-derived instance
        
        :returns:  If :code:`key` is an id or (source,external_id) pair, all rows of the \
        corresponding pattern will be returned.  If :code:`key` is a \
        backing-store-specific :py:class:`zounds.model.frame.Address`-derived \
        instance, all rows specified by the address will be returned.
        '''
        pass
    
    @abstractmethod
    def stat(self,feature,aggregate,axis = 0,step = 1):
        '''
        Compute aggregate statistics over all features in the data store
        
        :param feature: A :py:class:`~zounds.model.frame.Feature`-derived \
        instance defined on the current application's \
        :py:class:`~zounds.model.frame.Frames`-derived instance, or a string \
        corresponding to a feature's key.
        
        :param aggregate: the aggregate function, e.g., sum, mean, max, etc...
        
        :param axis:  The axis over which the aggregate function will be \
        computed
        
        :param step: The step size between collected rows.  If the feature's \
        absolute step size is greater than one, redundant data is stored, so \
        it's ok to only sample every n frames.
        
        :returns: a numpy array representing the aggregate value
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
        Iterate over a feature from the pattern with :code:`_id`
        
        :param _id: a zounds id
        
        :param feature: a feature key or :py:class:`~zounds.model.frame.Feature` \
        derived instance.
        
        :param step: the frequency at which samples should be drawn from the \
        available frames.
        
        :param chunksize: if greater than one, chunks of feature values are \
        returned.  E.g., if :code:`step` is 1 and :code:`chunksize` is 10, chunks \
        of 10 frames will be returned. If :code:`step` is 2 and :code:`chunksize` \
        is 10, chunks of 5 frames will be returned.
        
        :returns: an iterator over :code:`feature`  
        
        '''
        pass
  
    @abstractmethod
    def get_features(self):
        '''
        
        :returns: a dictionary mapping feature keys to the current set of \
        :py:class:`~zounds.model.frame.Feature`-derived instances defined \
        on this application's :py:class:`~zounds.model.frame.Frames`-derived \
        class.
        
        '''
        pass
    
    
    @abstractmethod
    def get_dtype(self,key):
        '''
        Get the datatype of a feature
        
        :param key: may be a :py:class:`~zounds.model.frame.Feature` instance, \
        or a string
        
        :returns: The :code:`numpy.dtype` of the feature 
        '''
        pass
    
    @abstractmethod
    def get_dim(self,key):
        '''
        Get the shape of a single frame of a feature.  E.g., the shape of an fft
        feature in a zounds application with a window size of 2048 would be
        :code:`(1024,)`
        
        :param key: may be a :py:class:`~zounds.model.frame.Feature` instance, \
        or a string
        
        :returns: A tuple representing the shape of a single frame of the \
        feature
        '''
        pass
    
    @abstractmethod
    def iter_all(self,step = 1):
        '''
        Iterate over *all* frames in the database, returning two-tuples of 
        (:py:class:`~zounds.model.frame.Address`, :py:class:`~zounds.model.frame.Frames`).
        
        :param step: An integer representing the step size of the iterator
        
        :returns: An iterator which will yield two-tuples of \
        (:py:class:`~zounds.model.frame.Address`, :py:class:`~zounds.model.frame.Frames`)
        '''
        pass
    
    @abstractmethod
    def iter_id(self,_id,chunksize,step = 1):
        '''
        Iterate over the frames from a single id, returning two-tuples of 
        (:py:class:`~zounds.model.frame.Address`, :py:class:`~zounds.model.frame.Frames`).
        
        :param _id: the zounds id of the pattern to iterate over
        
        :param chunksize: if greater than one, chunks of rows are returned.  \
        E.g., if :code:`step` is 1 and :code:`chunksize` is 10, chunks \
        of 10 frames will be returned. If :code:`step` is 2 and :code:`chunksize` \
        is 10, chunks of 5 frames will be returned.
        
        :param step: the step size of the iterator
        
        :returns: An iterator which will yield two-tuples of \
        (:py:class:`~zounds.model.frame.Address`, :py:class:`~zounds.model.frame.Frames`) \
        from a single zounds pattern
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