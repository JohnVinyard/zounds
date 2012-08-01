import numpy as np

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
    end = l - windowsize
    if l <= windowsize:
        return  windowsize
    elif l % stepsize:
        left = l%stepsize
        diff = stepsize - left
        return l + diff
    else:
        return l

def _wcut(l,windowsize,stepsize):
    end = l - windowsize
    if l <= windowsize:
        return 0
    elif end % stepsize:
        return windowsize + (((end//stepsize))*stepsize)
    else:
        return l
        
         
def windowed(a,windowsize,stepsize = None,dopad = False):
    l = a.shape[0]
    if stepsize is None:
        stepsize = windowsize
    if dopad:
        a = pad(a,_wpad(l,windowsize,stepsize))
        leftover = np.zeros(0)
    else:
        c = _wcut(l,windowsize,stepsize)
        leftover = a[c:]
        a = a[:c]
        
    print a.shape
    n = 1+(l-windowsize)//(stepsize)
    s = a.strides[0]
    newshape = (n,windowsize)+a.shape[1:]
    print newshape
    newstrides = (stepsize*s,s) + a.strides[1:]
    return leftover,np.ndarray.__new__(\
            np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)     
        
    
import numpy as N

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is 
    unevenly strided and being flattened or because end is set to 
    'pad' or 'wrap').
    """

    if axis is None:
        a = N.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = N.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b
        
        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis]+(n,length)+a.shape[axis+1:]
    newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]

    try: 
        return N.ndarray.__new__(N.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]
        return N.ndarray.__new__(N.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)