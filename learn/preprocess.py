from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

from nputil import safe_unit_norm as sun


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

class Add(Preprocess):
    '''
    Add a number to every element in the input
    '''
    
    def __init__(self,n):
        Preprocess.__init__(self)
        self.n = n
    
    def _preprocess(self,data):
        return data + self.n
    
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

class UnitNorm(Preprocess):
    
    def __init__(self):
        Preprocess.__init__(self)
        
    def _preprocess(self,data):
        return sun(data)


class PreprocessBarkBands(MeanStd):
    
    def __init__(self, mean = None, std = None, axis = 0):
        MeanStd.__init__(self,mean = mean, std = std, axis = axis)
        
    def _preprocess(self,data):
        data += 1
        data = np.log(data)
        return MeanStd._preprocess(self, data)
    
        
class SequentialPreprocessor(Preprocess):
    '''
    Apply a chain of preprocessors, in order.
    '''
    def __init__(self,preprocessors):
        Preprocess.__init__(self)
        self._p = preprocessors
        
    def _preprocess(self,data):
        for p in self._p:
            data = p(data)
        return data
    
    
