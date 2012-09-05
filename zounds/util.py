from __future__ import division
import numpy as np
import os.path
from multiprocessing import Process



# KLUDGE: There have been a couple instances (the multiprocess 
# FileSystemFrameController.sync() method being the most recent), where 
# multiprocessing.Pool.map() just starts, and hangs. No exceptions are thrown,
# and the program does not terminate.  New python processes are created, but
# they sit and do nothing.  I've found that invoking multiprocess.Process
# directly avoids this problem.
class PoolX(object):
    
    def __init__(self,nprocesses):
        object.__init__(self)
        self._nprocesses = nprocesses
    
    def map(self,target,args):
        for i in range(0,len(args),self._nprocesses):
            argchunk = args[i : i + self._nprocesses]
            procs = [Process(target = target, args = a) for a in argchunk]
            [p.start() for p in procs]
            [p.join() for p in procs]


def ensure_path_exists(filename_or_directory):
    '''
    Given a filename, ensure that the path to it exists
    '''
    
    if not filename_or_directory:
        raise ValueError(\
                'filename_or_directory must be a path to a file or directory')
    
    parts = os.path.split(filename_or_directory)
    # find out if the last part has a file extension
    subparts = os.path.splitext(parts[-1])
    extension = subparts[-1]
    # we're only interested in creating directories, so leave off the last part,
    # if it's a filename
    parts = parts[:-1] if extension else parts
    path = os.path.join(*parts)
    
    
    if path:
        try: 
            os.makedirs(path)
        except OSError:
            # This probably means that the path already exists
            pass


# TODO: This should go into a new "synthesize" module
def testsignal(hz,seconds=5.,sr=44100.):
    '''
    Create a sine wave at hz for n seconds
    '''
    # cycles per sample
    cps = hz / sr
    # total samples
    ts = seconds * sr
    return np.sin(np.arange(0,ts*cps,cps) * (2*np.pi))

# TODO: This should go into a new synthesize module.
def notes(events,envelope,sr=44100.):
    '''
    events   - a list of tuples of (time_secs,pitch)
    envelope - an envelope to be applied to each note. All notes will have 
               the same duration. 
    '''
    # sort the events by ascending time
    srt = sorted(events,cmp = lambda e1,e2 : cmp(e1[0],e2[0]))
    # the length of the envelope (and therefore the lenght of all events), in
    # samples
    le = len(envelope)
    # the length of the entire signal
    l = (srt[-1][0] * sr) + le
    sig = np.zeros(l)
    # the length of the envelope, in seconds
    els = le/sr
    for e in srt:
        ts = int(e[0] * sr)
        note = testsignal(e[1],els) * envelope
        sig[ts : ts + le] += note
    return sig
        


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