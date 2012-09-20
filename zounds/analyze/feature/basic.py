from __future__ import division
import numpy as np

from scipy.signal import convolve

from zounds.analyze.extractor import SingleInput
from zounds.nputil import safe_log,safe_unit_norm as sun,norm_shape,pack,flatten2d


class Basic(SingleInput):
    
    # TODO: Can inshape be inferred from the feature on which we rely?
    def __init__(self, inshape = None, outshape = None, op = None, needs = None, \
                  key = None, nframes = 1, step = 1):
        
        SingleInput.__init__(\
                self,needs = needs, key = key, nframes = nframes, step = step)
        
        
        self._inshape = norm_shape(inshape)
        self._outshape = norm_shape(outshape)
        self._op = op
        
    def dim(self,env):
        return self._outshape
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        data = self.in_data
        l = data.shape[0]
        data = data.reshape((l,) + self._inshape)
        return self._op(data).reshape((l,) + self._outshape)


class Abs(Basic):
    '''
    Give input data absolute value
    '''
    
    def __init__(self, inshape = None, needs = None, key = None, 
                 nframes = 1, step = 1):
        '''__init__
        
        :param inshape: The shape of input data.  The output data will be the \
        same shape
        
        :param needs: The :py:class:`~zounds.model.frame.Feature` to give \
        absolute value to
        
        :param nframes: The number of frames of data required from the input \
        feature to perform a computation.
        
        :param step: The number of frames of the input feature described by a \
        single frame of this feature 
        '''
        
        Basic.__init__(self,inshape = inshape, outshape = np.product(inshape),
                       op = np.abs, needs = needs, key = key, 
                       nframes = nframes, step = step)

class UnitNorm(Basic):
    '''
    Give the input data unit-norm
    '''
    
    def __init__(self, inshape = None, needs = None, key = None, 
                 nframes = 1, step = 1):
        '''__init__
        
        :param inshape: The shape of input data.  The output data will be the \
        same shape
        
        :param needs: The :py:class:`~zounds.model.frame.Feature` to give \
        unit-norm to
        
        :param nframes: The number of frames of data required from the input \
        feature to perform a computation.
        
        :param step: The number of frames of the input feature described by a \
        single frame of this feature 
        '''
        
        Basic.__init__(self, inshape = inshape, outshape = np.product(inshape),
                       op = sun, needs = needs, key = key,
                       nframes = nframes, step = step)
class Log(Basic):
    '''
    Take the log of the input
    '''
    
    def __init__(self, inshape = None, needs = None, key = None,
                 nframes = 1, step = 1):
        '''__init__
        
        :param inshape: The shape of input data.  The output data will be the \
        same shape
        
        :param needs: The :py:class:`~zounds.model.frame.Feature` to take the \
        log of
        
        :param nframes: The number of frames of data required from the input \
        feature to perform a computation.
        
        :param step: The number of frames of the input feature described by a \
        single frame of this feature 
        '''
        
        Basic.__init__(self, inshape = inshape, outshape = np.prod(inshape),
                       op = safe_log, needs = needs, key = key,
                       nframes = nframes, step = step)


class SliceX(SingleInput):
    '''
    Output a slice of the input
    '''
    
    # TODO: Why the hell am I not just passing a slice instance, instead of
    # a tuple which is used to instantiate a slice?
    def __init__(self, needs = None, key = None, slce = None, 
                 nframes = 1, step = 1):
        '''__init__
        
        :param needs: The :py:class:`~zounds.model.frame.Feature` to take the \
        log of
        
        :param slce: :code:`slice` can be one of:
            
            * a tuple of length one. This is interpreted as the ending value of the slice
            * a tuple of length two. These are interpreted as the start and stop values of the slice
            * a tuple of length three. These are interpreted as the start, stop, and step values of the slice
        
        :param nframes: The number of frames of data required from the input \
        feature to perform a computation.
        
        :param step: The number of frames of the input feature described by a \
        single frame of this feature
        '''
        
        SingleInput.__init__(self, needs = needs, key = key, 
                             nframes = nframes, step = step)
        if 1 == len(slce):
            self._slice = slice(0,slce[0],1)
        elif 2 == len(slce):
            self._slice = slice(slce[0],slce[1],1)
        else:
            self._slice = slice(*slce)
    
    def dim(self,env):
        return int((self._slice.stop - self._slice.start) / self._slice.step)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return flatten2d(self.in_data)[:,self._slice]


class Pack(SingleInput):
    
    def __init__(self, nbits = None,needs = None, key = None, nframes = 1, step = 1):
        if nbits not in [32,64]:
            raise ValueError('nbits must be one of (32,64)')
        
        self._dim = ()
        # TODO: use the mappings defined in nputil
        self._dtype = np.uint32 if nbits == 32 else np.uint64
        SingleInput.__init__(\
                self,needs = needs, key = key, nframes = nframes, step = step)
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return self._dtype
    
    def _process(self):
        return pack(self.in_data)
        


class Threshold(SingleInput):
    
    def __init__(self,needs = None, key = None, thresh = None, nframes = 1, step = 1):
        SingleInput.__init__(self,needs = needs, key = key, nframes = nframes, step = step)
        self._thresh = thresh
        try:
            self._dim = len(thresh)
        except AttributeError:
            self._dim = ()
    
    @property
    def dtype(self):
        return np.uint8
    
    def dim(self,env):
        return self._dim
    
    def _process(self):
        data = flatten2d(self.in_data)
        return (data > self._thresh).astype(self.dtype)

# KLUDGE: This doesn't belong in this module
class Sharpen(SingleInput):
    '''
    Convolve input with a simple high-pass filter to accentuate peaks
    '''
    
    def __init__(self,needs = None, key = None, nframes = 1, step = 1, 
                 kernel_size = 11,sharpness = 10,inshape = None):
        SingleInput.__init__(self,needs = needs, nframes = nframes,
                             step = step, key = key)
        
        if not kernel_size % 2:
            raise ValueError('kernel_size must be odd')
        
        self._filter = -np.ones(kernel_size)
        self._filter[int(kernel_size / 2)] = sharpness
        self._inshape = norm_shape(inshape)
    
    def dim(self,env):
        return self._inshape
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        data = self.in_data
        l = data.shape[0]
        data = data.reshape((l,) + self._inshape)
        out = np.zeros(data.shape)
        for i,d in enumerate(data):
            out[i] = convolve(d,self._filter,mode = 'same')
        return out
        