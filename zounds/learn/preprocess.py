from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

from zounds.nputil import safe_unit_norm as sun


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

class Multiply(Preprocess):
    
    def __init__(self,n):
        Preprocess.__init__(self)
        self.n = n
    
    def _preprocess(self,data):
        return data * self.n

class SubtractMean(Preprocess):
    
    def __init__(self,mean = None, axis = 0):
        '''
        mean - If provided, this value will be subtracted from all data. A mean
               will never be computed from training data.
        axis - The axis along which the mean will be computed.  Generally, zero
               means that an average will be computed feature-wise, while one
               means that an average will be computed example-wise.
        '''
        Preprocess.__init__(self)
        self._mean = mean
        self._axis = axis
        # KLUDGE: This code is duplicated in DivideByStd, but I'm not quite
        # sure where to factor it out to, or if there's a more "standard"
        # numpy way of doing this
        self._sl = [slice(None)] + ([np.newaxis] * axis)
    
    @property
    def mean(self):
        return self._mean[self._sl]
    
    def _preprocess(self,data):
        if self._mean is None:
            self._mean = data.mean(self._axis)
        
        return data - self.mean

class DivideByStd(Preprocess):
    
    def __init__(self,std = None, axis = 0):
        '''
        std - If provided, this value will be subtracted from all data. A
              standard deviation value will never be computed from the training
              data.
        axis - The axis along which the mean will be computed.  Generally, zero
               means that an average will be computed feature-wise, while one
               means that an average will be computed example-wise.
        '''
        Preprocess.__init__(self)
        self._std = std
        self._axis = axis
        # KLUDGE: This code is duplicated in SubtractMean, but I'm not quite
        # sure where to factor it out to, or if there's a more "standard"
        # numpy way of doing this
        self._sl = [slice(None)] + ([np.newaxis] * axis)
    
    @property
    def std(self):
        return self._std[self._sl]
    
    def _preprocess(self,data):
        if self._std is None:
            self._std = data.std(self._axis)
        
        return data / self.std
    

class UnitNorm(Preprocess):
    
    def __init__(self):
        Preprocess.__init__(self)
        
    def _preprocess(self,data):
        return sun(data)
    
    
class SequentialPreprocessor(Preprocess):
    '''
    Apply a chain of preprocessors, in order.
    '''
    def __init__(self,preprocessors):
        Preprocess.__init__(self)
        self._p = preprocessors
    
    def __getitem__(self,index):
        return self._p[index]
        
    def _preprocess(self,data):
        for p in self._p:
            data = p(data)
        return data

class MeanStd(SequentialPreprocessor):
    
    def __init__(self,mean = None, std = None, axis = 0):
        SequentialPreprocessor.__init__(\
                        self,
                        [SubtractMean(mean = mean, axis = axis),
                         DivideByStd(std = std, axis = axis)])
    @property
    def mean(self):
        return self._p[0]._mean
    
    @property
    def std(self):
        return self._p[1]._std

class PreprocessBarkBands(MeanStd):
    
    def __init__(self, mean = None, std = None, axis = 0):
        MeanStd.__init__(self,mean = mean, std = std, axis = axis)
        
    def _preprocess(self,data):
        data += 1
        data = np.log(data)
        return MeanStd._preprocess(self, data)
    

class Whiten(Preprocess):
    
    def __init__(self,weights = None):
        Preprocess.__init__(self)
        self._weights = None
    
    def _preprocess(self,data):
        if self._weights is None:
            cov = np.dot(data.T,data)
            u,s,v = np.linalg.svd(cov)
            d = np.diag(1. / np.sqrt(np.diag(s) + 1e-12))
            self._weights = np.dot(d,u.T)
        
        return data * self._weights
        

    
    
