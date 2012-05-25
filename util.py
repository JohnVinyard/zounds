from __future__ import division
import numpy as np
import os.path

# TODO: Get rid of this file! Move pad into a more appropriate location

def pad(a,desiredlength):
    '''
    Pad an n-dimensional numpy array with zeros so that it is the
    desired length.  Return it unchanged if it is greater than or
    equal to the desired length
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

def flatten2d(arr):
    ls = len(arr.shape)
    if 1 == ls:
        raise ValueError('Cannot turn 1d array into 2d array')
    elif 2 == ls:
        return arr
    else:
        return arr.reshape((arr.shape[0],np.product(arr.shape[1:])))

def downsampled_shape(shape,factor):
    if 2 != len(shape):
        raise ValueError('downsampling can only be performed on 2d arrays')
    
    return int(shape[0] / factor),int(shape[1] / factor)

def downsample(myarr,factor):
    '''
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.
    '''
    if 1 == factor:
        return myarr
    
    xs,ys = myarr.shape
    crarr = myarr[:xs-(xs % int(factor)),:ys-(ys % int(factor))]
    dsarr = np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]).mean(axis=0)
    return dsarr

def downsample3d(arr,factor):
    if 1 == factor:
        return arr
    
    oldshape = np.array(arr.shape)
    newshape = np.array(oldshape / factor).astype(np.int16)
    newarr = np.zeros(newshape)
    for x in range(0,newshape[0]):
        for y in range(0,newshape[1]):
            for z in range(0,newshape[2]):
                xstart = x*factor
                xstop = xstart + factor
                ystart = y*factor
                ystop = ystart + factor
                zstart = z * factor
                zstop = zstart + factor
                newarr[x,y,z] = arr[xstart : xstop, ystart : ystop, zstart : zstop].mean()
    return newarr
                
    
    
    

def ensure_path_exists(filename):
    '''
    Given a filename, ensure that the path to it exists
    '''
    parts = os.path.split(filename)
    path = os.path.join(*parts[:-1])
    
    if path:
        try: 
            os.makedirs(path)
        except OSError:
            # This probably means that the path already exists
            pass

def testsignal(hz,seconds=5.,sr=44100.):
    # cycles per sample
    cps = hz / sr
    # total samples
    ts = seconds * sr
    return np.sin(np.arange(0,ts*cps,cps) * (2*np.pi))


# TODO: This is used in analyze.extractor and model.frame. Can it be
# factored out into a *better*, common location?
def recurse(fn):
    '''
    For classes with a nested, tree-like structure, whose nodes
    are of the same class, or at least implement the same interface,
    this function can be used as a decorator which will perform 
    a depth-first flattening of the tree, e.g.
    
    class Node:
    
        @recurse
        def descendants(self):
            return self.children
    '''
    def wrapped(inst,accum=None):
        if accum == None:
            accum = []
        s = fn(inst)
        funcname = fn.__name__
        try:
            accum.extend(s)
            for q in s:
                getattr(q,funcname)(accum)
        except TypeError:
            # the object was not iterable
            accum.append(s)
        
        # We don't want to return any node more than once
        return list(set(accum))
    
    return wrapped 
        
def sort_by_lineage(class_method):
    '''
    Return a function that will compare two objects of or
    inherited from the same class based on their ancestry
    '''
    def _sort(lhs,rhs):
        # the lineages of the left and right hand sides
        lhs_l = class_method(lhs)
        rhs_l = class_method(rhs)
        
        if lhs in rhs_l and rhs in lhs_l:
            raise ValueError('lhs and rhs are ancestors of each other')
        
        if rhs in lhs_l:
            # lhs depends on rhs, directly or indirectly
            return 1
        
        if lhs in rhs_l:
            # rhs depends on lhs, directly or indirectly
            return -1
        
        
        rhs_l_len = len(rhs_l)
        lhs_l_len = len(lhs_l)
        
        if rhs_l_len < lhs_l_len:
            # rhs has fewer dependencies than lhs
            return 1
        
        if lhs_l_len < rhs_l_len:
            # lhs has fewer dependencies than rhs
            return -1
        
        # lhs and rhs have no direct relationship, and have the same number
        # of dependencies
        return 0
    
    return _sort
    

    
