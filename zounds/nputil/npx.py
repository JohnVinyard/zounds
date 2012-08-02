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
        return np.zeros(0),a
    
    if windowsize == 1 and stepsize > 1:
        return np.zeros(0),a[::stepsize]
    
    # the original length of the input array
    l = a.shape[0]
    
    if dopad:
        # pad the array with enough zeros so that there are no leftover samples
        a = pad(a,_wpad(l,windowsize,stepsize))
        # no leftovers; an empty array
        leftover = np.zeros(0)
    else:
        # cut the array so that any leftover samples are returned seperately
        c = _wcut(l,windowsize,stepsize)
        leftover = a[c:]
        a = a[:c]
    
    if 0 == a.shape[0]:
        return leftover,np.zeros(0)

    
    n = 1+(a.shape[0]-windowsize)//(stepsize)
    s = a.strides[0]
    newshape = (n,windowsize)+a.shape[1:]
    newstrides = (stepsize*s,s) + a.strides[1:]
    return leftover,np.ndarray.__new__(\
            np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)