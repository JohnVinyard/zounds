from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np


from zounds.nputil import safe_unit_norm as sun,norm_shape,downsample,flatten2d
from zounds.util import tostring


class Preprocess(object):
    '''
    Preprocess is an abstract base class.  Derived classes must implement the
    :code:`_preprocess` method which should transform data in a manner appropriate
    to the learning algorithm being used. 
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)

    @abstractmethod    
    def _preprocess(self,data):
        '''_preprocess
        
        Transform data in a manner appropriate to the learning algorithm being
        used
        
        :param data: A numpy array whose first dimension represents examples, and \
        whose subsequent dimensions represent features
        '''
        raise NotImplemented()
    
    def __call__(self,data):
        return self._preprocess(data)
    
    def __str__(self):
        return tostring(self)
    
    def __repr__(self):
        return tostring(self)
    
    
class NoOp(Preprocess):
    '''
    Returns data unaltered.  Use when no preprocessing is required.
    '''
    
    def __init__(self):
        Preprocess.__init__(self)
         
    def _preprocess(self,data):
        return data


# TODO: Don't save the mean if axis = 1
class SubtractMean(Preprocess):
    '''
    Subtract the mean of the data from the data itself, either feature or 
    example-wise.  This is intended to center features around zero.
    '''
    
    def __init__(self,mean = None, axis = 0):
        '''__init__
        
        :param mean:  If provided, this value will be subtracted from all data. \
        A mean will never be computed from training data.  Otherwise, a mean will \
        be computed the first time this preprocessor is called.  The mean computed \
        will be used for the lifetime of this instance.
        
        :param axis: The axis along which the mean will be computed.  Zero \
        (the default) means that an average will be computed feature-wise, \
        while one means that an average will be computed example-wise.
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

    def __repr__(self):
        return tostring(self,axis = self._axis)
    
    def __str__(self):
        return self.__repr__()

# TODO: Don't save the std if axis = 1
class DivideByStd(Preprocess):
    '''
    Divide data by its own standard deviation, either feature or example-wise. This
    is intended to give all features (or examples) equal variance.
    '''
    
    def __init__(self,std = None, axis = 0):
        '''__init__
        
        :param std: If provided, this value will be subtracted from all data. A \
        standard deviation value will never be computed from the training data.  \
        Otherwise, a standard deviation will be computed from the data the first \
        time this instance is called, and the value will be used for the lifetime \
        of this instance.
        
        :param axis: The axis along which the standard deviation will be computed.  \
        Zero (the default) means that standard deviation will be computed \
        feature-wise, while one means that standard deviation will be computed \
        example-wise.
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

    def __repr__(self):
        return tostring(self,axis = self._axis)
    
    def __str__(self):
        return self.__repr__()
    

class UnitNorm(Preprocess):
    '''
    Give all examples unit-norm.
    '''
    
    def __init__(self):
        Preprocess.__init__(self)
        
    def _preprocess(self,data):
        return sun(data)


class Abs(Preprocess):
    
    def __init__(self):
        Preprocess.__init__(self)
    
    def _preprocess(self,data):
        return np.abs(data)
    
    
class SequentialPreprocessor(Preprocess):
    '''
    Apply a chain of preprocessors, in order.
    '''
    
    def __init__(self,preprocessors):
        '''__init__
        
        :param preprocessors: A list of :py:class:`Preprocess`-derived classes \
        to be applied to data, in the order given.
        '''
        
        Preprocess.__init__(self)
        self._p = preprocessors
    
    def __getitem__(self,index):
        return self._p[index]
        
    def _preprocess(self,data):
        for p in self._p:
            data = p(data)
        return data
    
    def __repr__(self):
        return tostring(self,short = False,preprocessors = self._p)

class MeanStd(SequentialPreprocessor):
    '''
    A :py:class:`SequentialPreprocessor` that first subtracts the data's mean
    from itself, either feature or example-wise, and then divides the data by
    its standard deviation, either feature or example-wise.
    '''
    
    def __init__(self,mean = None, std = None, axis = 0):
        '''__init__
        
        :param mean: If :code:`None`, the mean will be computed from the data, \
        otherwise, the provided value will be subtracted from the data.
        
        :param std: If :code:`None`, the standard deviation will be computed from \
        the data, otherwise, the data will be divided by the value provided.
        
        :param axis:  If zero, data will be normalized feature-wise. Otherwise, \
        data will be normalized example-wise.
        '''
        
        # TODO: The axis *must* be zero if either mean or std are provided, since
        # there's no way for a client to knwo the number of examples of data that
        # will be passed to _preprocess()
        
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
        
class Downsample(Preprocess):
    '''
    Downsample data by a constant factor in every dimension.
    
    Downsample always expects to receive data as a two-dimensional array; the
    first dimension represents examples, and the second represents features.  
    Concretely, here's what it would look like to downsample 50 examples of
    10x10 rectangles, by a factor of 2::
    
        >>> data = np.ones((50,10,10))
        >>> data = data.reshape((50,100))
        >>> ds = Downsample((10,10),2)
        >>> ds(data).shape
        (50,25)
    
    The output is 50 5x5 rectangles, flattened into one dimension.
    '''
    
    
    def __init__(self,shape,factor,method = np.mean):
        '''__init__
        
        :param shape: A tuple representing the shape each example should be in \
        prior to downsampling
        
        :param factor: The factor to downsample by, in all dimensions
        
        :param method: The method by which "blocks" of values will be reduced to \
        a single value, e.g. mean, max, sum, etc...
        '''
        if not isinstance(factor,int):
            raise ValueError('factor must be an int')
    
        
        Preprocess.__init__(self)
        self._shape = norm_shape(shape)
        self._factor = factor
        self._method = method
    
    def _preprocess(self,data):
        # the shape the data should be in prior to downsampling
        realshape = (data.shape[0],) + self._shape
        # the factor tuple, which will cause downsampling in all but the first
        # (example-wise) dimension
        factor = (self._factor,) * len(self._shape)
        ds = downsample(data.reshape(realshape),factor, method = self._method)
        if ds.ndim == 1:
            return ds
        
        return flatten2d(ds)
    
    def __repr__(self):
        return tostring(self,factor = self._factor,method = self._method)
    

class Add(Preprocess):
    '''
    Add a number to every element in the input
    '''
    
    def __init__(self,n):
        Preprocess.__init__(self)
        self.n = n
    
    def _preprocess(self,data):
        return data + self.n
    
    def __repr__(self):
        return tostring(self,add = self.n)
    
    def __str__(self):
        return self.__repr__()

class Multiply(Preprocess):
    
    def __init__(self,n):
        Preprocess.__init__(self)
        self.n = n
    
    def _preprocess(self,data):
        return data * self.n
    
    def __repr__(self):
        return tostring(self,add = self.n)
    
    def __str__(self):
        return self.__repr__()
