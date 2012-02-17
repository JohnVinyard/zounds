from __future__ import division
import numpy as np

# TODO: Get rid of this file! Move pad into a more appropriate location

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



def testsignal(hz,seconds=5.,sr=44100.):
    # cycles per sample
    cps = hz / sr
    # total samples
    ts = seconds * sr
    return np.sin(np.arange(0,ts*cps,cps) * (2*np.pi))


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
        try:
            accum.extend(s)
            for q in s:
                getattr(q,fn.__name__)(accum)
        except TypeError:
            # the object was not iterable
            pass
        return accum
    
    return wrapped 
        
def sort_by_lineage(class_method,allow_circular = False):
    '''
    Return a function that will compare two objects of or
    inherited from the same class based on their ancestry
    '''
    def _sort(lhs,rhs):
        # the lineages of the left and right hand sides
        lhs_l = class_method(lhs)
        rhs_l = class_method(rhs)
        
        if lhs in rhs_l and rhs in lhs_l and not allow_circular:
            raise ValueError('lhs and rhs are ancestors of each other')
        
        if rhs in lhs_l:
            return 1
        
        if lhs in rhs_l:
            return -1
        
        return 0
    
    return _sort
    

    
