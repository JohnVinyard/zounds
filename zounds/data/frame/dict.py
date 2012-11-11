from frame import FrameController

class DictFrameController(FrameController):
    '''
    Completely useless, except for testing
    '''
    
    def __init__(self,framesmodel):
        FrameController.__init__(self,framesmodel)
        
    
    def concurrent_reads_ok(self):
        raise NotImplementedError
    
    def concurrent_writes_ok(self):
        raise NotImplementedError
        
    def sync(self,add,update,delete,chain):
        raise NotImplementedError
    
    def append(self,frames):
        raise NotImplementedError
    
    def get(self,key):
        raise NotImplementedError
  
    def get_features(self):
        raise NotImplementedError
      
    def set_features(self):
        raise NotImplementedError
    
    def get_dtype(self,key):
        raise NotImplementedError
    
    def get_dim(self,key):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def external_id(self,_id):
        raise NotImplementedError
    
    def list_ids(self):
        raise NotImplementedError
    
    def iter_feature(self,_id,feature_name,step = 1,chunksize = 1):
        raise NotImplementedError
    
    def exists(self,source,external_id):
        raise NotImplementedError
    
    def iter_all(self):
        raise NotImplementedError
    
    def iter_id(self):
        raise NotImplementedError
    
    def stat(self,feature,aggregate,axis = 0, step = 1):
        raise NotImplemented()
    
    def update_index(self):
        raise NotImplemented()
        
    def list_external_ids(self):
        raise NotImplemented()
    
    def address(self,_id):
        raise NotImplemented()
    
    def pattern_length(self,_id):
        raise NotImplemented()