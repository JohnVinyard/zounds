from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from bitarray import bitarray
import pyximport
pyximport.install()
from toeplitz import *


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
    
    Parameters
        shape - an int, or a tuple of ints
    
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
    
    raise TypeError('shape must be an int, or a tuple of ints')

def flatten2d(arr):
    ls = len(arr.shape)
    if 1 == ls:
        raise ValueError('Cannot turn 1d array into 2d array')
    elif 2 == ls:
        return arr
    else:
        return arr.reshape((arr.shape[0],np.product(arr.shape[1:])))


def downsampled_shape(shape,factor):
    '''
    Return the new shape of an array with shape, once downsampled
    by factor.
    '''
    return tuple((np.array(shape) / factor).astype(np.int32))


#def downsample(arr,factor,method = np.max, axes = None):
#    window_size = (factor,) * arr.ndim
#    axes = -np.arange(1,arr.ndim + 1)
#    windowed = sliding_window(arr,window_size,flatten = False)
#    return np.apply_over_axes(method, windowed, axes).squeeze()

def downsample(arr,factor,method = np.max):
    '''
    Downsample an n-dimensional array
    
    Parameters
        arr    - the array to be downsampled
        factor - the downsampling factor. If factor is an integer, it is assumed
                 that the array will be downsampled by a constant factor in every
                 dimension. If factor is a tuple, the array will be downsampled
                 by different factors in each dimension. If factor is a tuple 
                 whose size is less than the number of dimensions in arr, it
                 is assumed that the downsampling will only be applied to the
                 last len(factor) dimensions of arr.
        method - the method used to combine chunks into single values, e.g.
                 mean, or max.
    '''
    if isinstance(factor,int):
        # factor is an integer, so we'll downsample by a constant factor over
        # all dimensions of the input array
        factor = (factor,) * arr.ndim
        lf = arr.ndim
    else:
        # factor is a tuple, so we'll downsample over the last n dimensions of
        # arr by the factors specified
        factor = norm_shape(factor)
        lf = len(factor)
        if lf > arr.ndim:
            raise ValueError(\
            'The number of factors must be less than or equal to the number of dimensions in arr')
    
    # the axes over which to apply the reduction
    axes = -np.arange(1,lf + 1)
    # the window size in each dimension of arr
    window_size = ((1,) * (arr.ndim - lf)) + factor
    # get non-overlapping windows whose dimensions are specified by factor
    windowed = sliding_window(arr,window_size,flatten = False)
    # apply the reduction and remove any extraneous dimensions
    return np.apply_over_axes(method, windowed, axes).squeeze()
    
    
    
def safe_log(a):
    '''
    Return the element-wise log of an array, checking for negative
    array elements and avoiding divide-by-zero errors.
    '''
    if np.any([a < 0]):
        raise ValueError('array contains negative components')
    
    return np.log(a + 1e-12)



def safe_unit_norm(a):
    '''
    Ensure that the vector or vectors have unit norm
    '''
    if 1 == len(a.shape):
        n = np.linalg.norm(a)
        if n:
            return a / n
        return a
    
    norm = np.sum(np.abs(a)**2,axis=-1)**(1./2)
    # Dividing by a norm of zero will cause a warning to be issued. Set those
    # values to another number. It doesn't matter what, since we'll be dividing
    # a vector of zeros by the number, and 0 / N always equals 0.
    norm[norm == 0] = -1e12
    return a / norm[:,np.newaxis]



def pad(a,desiredlength):
    '''
    Pad an n-dimensional numpy array with zeros along the zero-th dimension
    so that it is the desired length.  Return it unchanged if it is greater 
    than or equal to the desired length
    '''
    
    if len(a) >= desiredlength:
        return a
    
    islist = isinstance(a,list)
    a = np.array(a)
    diff = desiredlength - len(a)
    shape = list(a.shape)
    shape[0] = diff
     
    padded = np.concatenate([a,np.zeros(shape,dtype=a.dtype)])
    return padded.tolist() if islist else padded


def _wpad(l,windowsize,stepsize):
    '''
    Parameters
        l - The length of the input array
        windowsize - the size of each window of samples
        stepsize - the number of samples to move the window each step
    
    Returns
        The length the input array should be so that no samples are leftover
    '''
    if l <= windowsize:
        return  windowsize
    
    nsteps = ((l // stepsize) * stepsize)
    overlap = (windowsize - stepsize)
    if overlap:
        return nsteps + overlap
     
    diff = (l - nsteps)
    left = max(0,windowsize - diff)
    return l + left if diff else l

def _wcut(l,windowsize,stepsize):
    '''
    Parameters
        l - The length of the input array
        windowsize - the size of each window of samples
        stepsize - the number of samples to move the window each step
    
    Returns
        The length the input array should be so that leftover samples are ignored
    '''
    end = l - windowsize
    if l <= windowsize:
        return 0,0
    elif end % stepsize:
        l = windowsize + ((end//stepsize)*stepsize)
    
    return l,l - (windowsize - stepsize)
        
 
def windowed(a,windowsize,stepsize = None,dopad = False):
    '''
    Parameters
        a          - the input array to restructure into overlapping windows
        windowsize - the size of each window of samples
        stepsize   - the number of samples to shift the window each step. If not
                     specified, this defaults to windowsize
        dopad      - If false (default), leftover samples are returned seperately.
                     If true, the input array is padded with zeros so that all
                     samples are used. 
    '''
    if windowsize < 1:
        raise ValueError('windowsize must be greater than or equal to one')
    
    if stepsize is None:
        stepsize = windowsize
    
    if stepsize < 1:
        raise ValueError('stepsize must be greater than or equal to one')
    
    if windowsize == 1 and stepsize == 1:
        # A windowsize and stepsize of one mean that no windowing is necessary.
        # Return the array unchanged.
        return np.zeros((0,) + a.shape[1:]),a
    
    if windowsize == 1 and stepsize > 1:
        return np.zeros(0),a[::stepsize]
    
    # the original length of the input array
    l = a.shape[0]
    
    if dopad:
        p = _wpad(l,windowsize,stepsize)
        # pad the array with enough zeros so that there are no leftover samples
        a = pad(a,p)
        # no leftovers; an empty array
        leftover = np.zeros((0,) + a.shape[1:])
    else:
        # cut the array so that any leftover samples are returned seperately
        c,lc = _wcut(l,windowsize,stepsize)
        leftover = a[lc:]
        a = a[:c]
    
    if 0 == a.shape[0]:
        return leftover,np.zeros(a.shape)

    
    n = 1+(a.shape[0]-windowsize)//(stepsize)
    s = a.strides[0]
    newshape = (n,windowsize)+a.shape[1:]
    newstrides = (stepsize*s,s) + a.strides[1:]
    return leftover,np.ndarray.__new__(\
            np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)


def sliding_window_1d(a,ws,ss = None):
    '''
    Parameters
        a  - a 1D array
        ws - the window size, in samples
        ss - the step size, in samples. If not provided, window and step size
             are equal. 
    '''
    
    if None is ss:
        # no step size was provided. Return non-overlapping windows
        ss = ws
    
    # calculate the number of windows to return, ignoring leftover samples, and
    # allocate memory to contain the samples
    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw,ws),dtype = a.dtype)
    
    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]
    
    return out
        

from itertools import product
def sliding_window_nd(a,ws,ss = None):
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    
    # ensure that window and step sizes are expressed as tuples, even if they're
    # one-dimensional
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    
    
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
    
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    
    # For each dimension, create a list of all valid slices
    slices = [[] for i in range(len(ws))]
    for i in xrange(len(ws)):
        for j in xrange(0,shape[i] - ws[i],ss[i]):
            slices[i].append(slice(j,j + ws[i]))
    # Get an iterator over all valid n-dimensional slices of the input
    allslices = product(*slices)
    
    # Allocate memory to hold all valid n-dimensional slices
    nslices = np.product([len(s) for s in slices])
    out = np.ndarray((nslices,) + tuple(ws),dtype = a.dtype)
    for i,s in enumerate(allslices):
        out[i] = a[s]
    
    return out
            
    
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
    
    Returns
        an array containing each n-dimensional window from a
    '''
    
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    
    
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
    
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
    
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


class TypeCodes(object):
    _bits = [8,16,32,64]
    _np_types = [np.uint8,np.uint16,np.uint32,np.uint64]
    _type_codes = ['B','H','L','Q']
    
    @classmethod
    def _fromto(cls,f,t,v):
        return t[f.index(v)]

    @classmethod
    def _whichlist(cls,v):
        if isinstance(v,int):
            return cls._bits
        
        if isinstance(v,str):
            return cls._type_codes
        
        if isinstance(v,type):
            return cls._np_types
        
        raise ValueError('%s is not a valid key' % v)
    
    @classmethod
    def bits(cls,v):
        return cls._fromto(cls._whichlist(v), cls._bits, v)
    
    @classmethod
    def np_dtype(cls,v):
        return cls._fromto(cls._whichlist(v), cls._np_types, v)
    
    @classmethod
    def type_code(cls,v):
        return cls._fromto(cls._whichlist(v), cls._type_codes, v)
        
    

def pack(a):
    '''
    Interpret an NxM numpy array as an N-length list of 32 or 64 bit numbers
    '''
    try:
        tc = TypeCodes.type_code(a.shape[1])
    except KeyError:
        raise ValueError('a must have a second dimension with shape 32 or 64')
    
    b = bitarray()
    b.extend(a.ravel())
    return np.fromstring(b.tobytes(),dtype = TypeCodes.np_dtype(tc))


class Packer(object):
    
    def __init__(self,totalbits,chunkbits = 64):
        try:
            self._typecode = TypeCodes.type_code(chunkbits)
        except KeyError:
            raise ValueError('chunkbits must be one of %s' % \
                             (str(TypeCodes._type_codes)))
        self._nchunks = int(np.ceil(totalbits / chunkbits))
        self._padding = (self._nchunks*chunkbits) - totalbits
        self._dtype = TypeCodes.np_dtype(self._typecode) 
    
    def allocate(self,l):
        '''
        Allocate memory for l packed examples
        '''
        return np.ndarray((l,self._nchunks),dtype = self._dtype)
    
    def __call__(self,a):
        l = a.shape[0]
        z = np.zeros((l,self._padding),dtype = a.dtype)
        padded = np.concatenate([a,z],axis = 1)
        b = bitarray()
        b.extend(padded.ravel())
        return np.fromstring(b.tobytes(),dtype = self._dtype)\
                .reshape((l,self._nchunks))
    

def hamming_distance(a,b):
    '''
    a - scalar
    b - array of scalars
    '''
    return count_bits(a^b)

def packed_hamming_distance(a,b):
    '''
    Interpret a as a "packed" scalar, i.e. an n-bit number where n may not be
    a power of 2. E.g., a 250-bit number would be represented by 4 64-bit integers.
    
    Interpret b as an array of "packed" scalars. Its second dimension should be
    the same length as a.
    '''
    xored = a ^ b
    bitcount = count_bits(xored.ravel())
    return bitcount.reshape(xored.shape).sum(1)

def jaccard_distance(a,b):
    '''
    a - scalar
    b - array of scalars
    '''
    intersection = count_bits(a & b)
    union = count_bits(a | b)
    return 1 - (intersection / union)

#from time import time
#if __name__ == '__main__':
#    a = np.random.binomial(1,.5,(10,250))
#    b = np.random.binomial(1,.5,(1000,250))
#    
#    start = time()
#    for z in a:
#        np.logical_xor(z,b).sum(1)
#    print 'original took %1.4f seconds' % (time()- start)
#    
#    a = np.random.random_integers(0,2**16,(10,5)).astype(np.uint64)
#    b = np.random.random_integers(0,2**16,(1000,5)).astype(np.uint64)
#    
#    m1 = None
#    m2 = None
#    
#    start = time()
#    for z in a:
#        xored = z ^ b
#        count_packed_bits(xored)
#    print 'cython took %1.4f seconds' % (time()- start)
#    
#    start = time()
#    xored = a[:,np.newaxis] ^ b
#    for x in xored:
#        count_packed_bits(x)
#    print 'xor outside loop took %1.4f seconds' % (time()- start)
#    
#    start = time()
#    out = np.ndarray((10,1000))
#    for i,z in enumerate(a):
#        xored = z ^ b
#        shape = xored.shape
#        bc = count_bits(xored.ravel())
#        out[i] = bc.reshape(shape).sum(1)
#    
#    print 'flat bit count loop took %1.4f seconds' % (time()- start)
    
if __name__ == '__main__':
    swd = sliding_window_dumb
    sw1d = sliding_window_1d
    
    a = np.zeros(100)
    b = swd(a,10)
    print b.shape
    b = swd(a,10,5)
    print b.shape
    
    b = sw1d(a,10)
    print b.shape
    b = sw1d(a,10,5)
    print b.shape
    
    a = np.zeros((10,10))
    b = swd(a,(2,2))
    print b.shape
    b = swd(a,(2,2),(1,1))
    print b.shape
    
    