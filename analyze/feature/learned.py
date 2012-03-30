from analyze.extractor import SingleInput
from model.pipeline import Pipeline


class Learned(SingleInput):
    '''
    A thin wrapper around a learned feature
    '''
    
    def __input__(self,
                  needs = None, 
                  nframes = 1, 
                  step = 1, 
                  key = None, 
                  pipeline_id = None,
                  dim = None,
                  dtype = None):
        
        
        SingleInput.__init__(\
                        self, needs=needs,nframes=nframes,step=step,key=key)
        self.pipeline = Pipeline[pipeline_id]
        self._dim = dim
        self._dtype = dtype
    
    
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return self._dtype
    
    def _process(self):
        data = self.in_data[0]
        return self.pipeline(data)
    
    
        
    