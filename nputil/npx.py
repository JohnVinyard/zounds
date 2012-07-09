import numpy as np


def safe_log(a):
    '''
    Return the element-wise log of an array, checking for negative
    array elements and avoiding divide-by-zero errors.
    '''
    if np.any([a < 0]):
        raise ValueError('array contains negative components')
    
    return np.log(a + 1e-12)

def _safe_unit_norm(a):
    '''
    Ensure that a vector has unit norm.
    '''
    n = np.linalg.norm(a)
    if n:
        return a / n
    
    return a

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