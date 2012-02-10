from __future__ import division
import numpy as np


# TODO: Write tests
def pad(a,desiredlength):
    '''
    Pad an n-dimensional numpy array with zeros so that it is the
    desired length.  Return it unchanged if it is greater than or
    equal to the desired length
    '''
    
    if len(a) >= desiredlength:
        return a
    
    
    diff = desiredlength - len(a)
    shape = list(a.shape)
    shape[0] = diff
    return np.concatenate([a,np.zeros(shape,dtype=a.dtype)])



def testsignal(hz,seconds=5.,sr=44100.):
    # cycles per sample
    cps = hz / sr
    # total samples
    ts = seconds * sr
    return np.sin(np.arange(0,ts*cps,cps) * (2*np.pi))


