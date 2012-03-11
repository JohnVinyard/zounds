from controller import Controller
from abc import ABCMeta,abstractmethod
from tables import openFile,IsDescription,StringCol,Int32Col,Col
import os.path
import re
import numpy as np

class FrameController(Controller):
    __metaclass__ = ABCMeta
    
    def __init__(self,framesmodel):
        Controller.__init__(self)
        self.model = framesmodel
    
     
    @abstractmethod
    def check(self,framesmodel):
        '''
        Returns true if the the FrameModel is in sync with the data store,
        i.e., all the features defined on the FrameModel with store = True
        are represented in the db
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
    def append(self,frames):
        '''
        Adds frames to the datastore
        '''
        pass
    
    @abstractmethod
    def get(self,indices,features=None):
        '''
        Gets rows, and optionally specific features from those rows.
        Indices may be a single index, a list of indices, or a slice.
        features may be a single feature or a list of them.
        '''
        pass
  
    @abstractmethod
    def get_features(self):
        '''
        Return the current set of features represented in the database
        '''
        pass
    
    @abstractmethod
    def set_features(self):
        '''
        Set the current set of features represented in the database
        '''
        pass
    
    @abstractmethod
    def get_dtype(self,key):
        '''
        Get the data type of the feature with key
        '''
        pass
    
    @abstractmethod
    def get_dim(self,key):
        '''
        Get the dimension of the feature with key
        '''
        pass


class PyTablesFrameController(FrameController):
    
    # TODO: How do I switch between read and write modes as necessary in a
    # transparent manner?
    
    def __init__(self,framesmodel,filepath):
        FrameController.__init__(self,framesmodel)
        self.filepath = filepath
        parts = os.path.split(filepath)
        self.filename = parts[-1]
        path = os.path.join(*parts[:-1])
        
        self.dbfile_write = None
        self.db_write = None
        
        self.dbfile_read = None
        self.db_read = None
        
        if len(parts) > 1:
            try:
                os.makedirs(path)
            except OSError:
                # This probably means that the path already exists
                pass
        
        
        if not os.path.exists(filepath):
            
            # KLUDGE: PyTables allows the creation of columns using a string
            # representation of a datatype, e.g. "float32", but not the numpy
            # type representing that type, e.g., np.float32.  This is a hackish
            # way of extracting a string representation that PyTables can
            # understand from a numpy type
            rgx = re.compile('\'(numpy\.)?(?P<type>[a-z0-9]+)\'')
            def get_type(np_dtype):
                m = rgx.search(str(np_dtype))
                if not m:
                    raise ValueError('Unknown dtype %s' % str(np_dtype))
                return m.groupdict()['type']
            
            self.dbfile_write = openFile(filepath,'w')
            
            # create the table's schema from the FrameModel
            # KLUDGE: This should be somehow determined by FrameModel also
            
            self.steps = {'source' : 1, '_id' : 1, 'framen' : 1}
            desc = {
                    'source' : StringCol(itemsize=20,pos = 0),
                    '_id'    : StringCol(itemsize=20,pos = 1),
                    'framen' : Int32Col(pos=2)
                    
                    }
            
            pos = len(desc)
            dim = self.model.dimensions()    
            for k,v in dim.iteritems(): 
                desc[k] = Col.from_type(get_type(v[1]),shape=v[0],pos=pos)
                self.steps[k] = v[2]
                pos += 1
                
            # create the table
            self.dbfile_write.createTable(self.dbfile_write.root, 'frames', desc)
            
            self.db_write = self.dbfile_write.root.frames
            
            # create indices on any column that we can
            for k,v in desc.iteritems():
                col = getattr(self.db_write.cols,k)
                oned = 2 == len(col.shape) and col.shape[1] == 1
                if isinstance(col,StringCol) or oned:
                    col.createIndex()
                    
            self.dbfile_write.close()
            

        self.dbfile_read = openFile(filepath,'r')
        self.db_read = self.dbfile_read.root.frames
        
        # create our buffer
        def lcd(numbers):
            i = 1
            while any([i % n for n in numbers]):
                i += 1
            return i
        self._desired_buffer_size = 100
        
        # find the lowest common multiple of all step sizes
        l = lcd(self.steps.values())
        # find a whole number multiple of the lowest common
        # multiple that puts us close to our desired buffer size
        self._buffer_size = l * int(self._desired_buffer_size / l)
        recarray_dtype = []
        for k in self.db_read.colnames:
            col = getattr(self.db_read.cols,k)
            recarray_dtype.append((k,col.dtype,col.shape))
        print recarray_dtype 
        self.buffer = np.recarray(self._buffer_size,dtype=recarray_dtype)
        self.buffer[:] = np.inf
        
    def append(self,frames):
        
        
        
        # switch to write mode
        self.close()
        self.dbfile_write = openFile(self.filepath,'a')
        self.db_write = self.dbfile_write.root.frames
        
        # TODO: Set a flag that lets everyone know we're writing
        # write the data
        
        # switch back to read mode
        self.close()
        self.dbfile_read = openFile(self.filepath,'r')
        self.db_read = self.dbfile_read.root.frames
        
    
    def close(self):
        if self.dbfile_write:
            self.dbfile_write.close()
        if self.dbfile_read:
            self.dbfile_read.close()
    
    def __del__(self):
        self.close()
        
    
    def check(self,framesmodel):
        raise NotImplemented()
    
    
    def sync(self,add,update,delete,chain):
        raise NotImplemented()
    
    
    
    
    
    def get(self,indices,features=None):
        raise NotImplemented()
  
    
    def get_features(self):
        raise NotImplemented()
    
    
    def set_features(self):
        raise NotImplemented()
    
    
    def get_dtype(self,key):
        raise NotImplemented()
    
    
    def get_dim(self,key):
        raise NotImplemented()
         
        

class DictFrameController(FrameController):
    
    def __init__(self,framesmodel):
        FrameController.__init__(self,framesmodel)
        
    def check(self,framesmodel):
        raise NotImplemented()
    
    
    def sync(self,add,update,delete,chain):
        raise NotImplemented()
    
    
    def append(self,frames):
        raise NotImplemented()
    
    
    def get(self,indices,features=None):
        raise NotImplemented()
  
    
    def get_features(self):
        raise NotImplemented()
    
    
    def set_features(self):
        raise NotImplemented()
    
    
    def get_dtype(self,key):
        raise NotImplemented()
    
    
    def get_dim(self,key):
        raise NotImplemented()
    
    
        
        