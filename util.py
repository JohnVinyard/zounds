from __future__ import division
from itertools import product
import numpy as np
import os.path
from constants import available_file_formats


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

def audio_files(path):
    '''
    Return the name of each sound file that Zounds can process in
    the given directory
    '''
    # list all files in the directory
    allfiles = os.listdir(path) 
    return filter(\
        lambda f : os.path.splitext(f)[1][1:] in available_file_formats,
        allfiles)


# TODO: Should this go into the nputil module as well?
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


def downsample(arr,factor,method = 'mean'):
    if method == 'mean':
        m = lambda a : np.mean(a)
    elif method == 'max':
        m = lambda a : np.mean(a)
    else:
        raise ValueError('method must be one of ("mean","max")')
    newshape = downsampled_shape(arr.shape,factor)
    newarr = np.zeros(newshape)
    prod = product(*[range(0,shape) for shape in newshape])
    for coord in prod:
        sl = [slice(x*factor,x*factor+factor) for x in coord]
        newarr[coord] = m(arr[sl])
    return newarr

def testsignal(hz,seconds=5.,sr=44100.):
    '''
    Create a sine wave at hz for n seconds
    '''
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
    

    
