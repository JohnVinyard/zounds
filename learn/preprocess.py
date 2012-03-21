from abc import ABCMeta,abstractmethod

class Preprocess(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)

    @abstractmethod    
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
        if self.mean is None:
            self.mean = data.mean(self.axis)
            
        newdata = data - self.mean
        
        if self.std is None:
            self.std = newdata.std(self.axis)
            
        newdata /= self.std
        
        return newdata
        
        

    
    
