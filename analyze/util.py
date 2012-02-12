from __future__ import division
import numpy as np


# TODO: Write tests
def pad(a,desiredlength):
    '''
    Pad an n-dimensional numpy array with zeros so that it is the
    desired length.  Return it unchanged if it is greater than or
    equal to the desired length
    '''
    islist = isinstance(a,list)
    
    if len(a) >= desiredlength:
        return a
    
    a = np.array(a)
    diff = desiredlength - len(a)
    shape = list(a.shape)
    shape[0] = diff
     
    padded = np.concatenate([a,np.zeros(shape,dtype=a.dtype)])
    return padded.tolist() if islist else padded
