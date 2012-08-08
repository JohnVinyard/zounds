import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from zounds.util import flatten2d


def norm_shape(shape):
    '''
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
    reg = a / norm[:,np.newaxis]
    reg[np.isnan(reg)] = 0
    return reg
    
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
    end = (((l - windowsize) // stepsize) * stepsize) + windowsize
    leftover = l - end
    return l + (stepsize - leftover) if leftover else l

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
        return 0
    elif end % stepsize:
        return windowsize + ((end//stepsize)*stepsize)
    else:
        return l
        
         
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
        # pad the array with enough zeros so that there are no leftover samples
        a = pad(a,_wpad(l,windowsize,stepsize))
        # no leftovers; an empty array
        leftover = np.zeros((0,) + a.shape[1:])
    else:
        # cut the array so that any leftover samples are returned seperately
        c = _wcut(l,windowsize,stepsize)
        leftover = a[c:]
        a = a[:c]
    
    if 0 == a.shape[0]:
        return leftover,np.zeros(a.shape)

    
    n = 1+(a.shape[0]-windowsize)//(stepsize)
    s = a.strides[0]
    newshape = (n,windowsize)+a.shape[1:]
    newstrides = (stepsize*s,s) + a.strides[1:]
    return leftover,np.ndarray.__new__(\
            np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)


# TODO: It might be handy to not always flatten the slices
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
    
    # convert ws, ss, and a.shape to tuples so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(ws),len(ss),len(shape)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
    
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    
    # how many slices will there be in each dimension
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
    