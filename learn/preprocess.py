
class Preprocess(object):
    
    def __init__(self):
        object.__init__(self)
    
    def _preprocess(self,data):
        raise NotImplemented()
    
    def __call__(self,data):
        return self._preprocess(data)
    
class NoOp(Preprocess):
     
    def __init__(self):
        Preprocess.__init__(self)
         
    def _preprocess(self,data):
        return data
    
class MeanStd(Preprocess):
    
    def __init__(self,mean = None, std = None, axis = 0):
        Preprocess.__init__(self)
        self.mean = mean
        self.std = std
        self.axis = axis
     
    def _preprocess(self,data):
        if not self.mean:
            self.mean = data.mean(self.axis)
            
        data -= self.mean
        
        if not self.std:
            self.std = data.std(self.axis)
            
        data /= self.std
        
        return data
        
        

    
    
