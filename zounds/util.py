from __future__ import division
import numpy as np
import os.path
from multiprocessing import Process
import argparse

# TODO: Can I use pprint.pformat here to get what I'm after in a more general
# and flexible way, e.g., for dictionaries? See Feature.__repr__ as an example
# of the problem
def _parts(name = '',nitems = 1,nesting = 1,short = True,delimiter = ',',brackets = ('(',')')):
    if short:
        j = '%s ' % delimiter
        tmpl = '%s%s%%s%s' % (name,brackets[0],brackets[1])
    else:
        tabs = '\t' * nesting
        j = '%s\n%s' % (delimiter,tabs)
        space = '\n' + tabs if nitems > 1 else ''
        tmpl = '%s%s%s%%s%s' % (name,brackets[0],space,brackets[1])  
    return j,tmpl


def tostring(obj,short = True,**kwargs):
    name = obj.__class__.__name__
    j,tmpl = _parts(nitems = len(kwargs),name = name,short = short)
    key_val_tmpl = '%s = %s'
    strs = []
    for k,v in kwargs.iteritems():
        if isinstance(v,list) or isinstance(v,tuple):
            iterj,itertmpl = _parts(\
                nitems = len(v),short = short, brackets = ('[',']'),nesting = 2)
            joined = iterj.join([str(a) for a in v])
            v = itertmpl % joined
        
        strs.append(key_val_tmpl % (k,v))
        
    attrs = j.join(strs)
    return tmpl % attrs
    

class SearchSetup(object):
    '''
    SearchSetup parses user-specified (at the command-line) search parameters,
    and builds a search to meet the user's specifications. 
    '''
    
    def __init__(self,framemodel):
        object.__init__(self)
        self._framemodel = framemodel
        self._parser = argparse.ArgumentParser()
    
    def _add_args(self):
        aa = self._parser.add_argument
        
        # required arguments
        aa('--feature',
           help='the name of the feature to use',
           required = True)
        aa('--searchclass',
           help='the name of the FrameSearch-derived class to use',
           required = True)
        aa('--sounddir',
           help='directory from which queries will be drawn',
           required = True)
        
        # optional arguments
        aa('--minseconds',
           help='the minimum length of a query',
           type=float,
           default = 1.0)
        aa('--maxseconds',
           help='the maximum length of a query',
           type=float,
           default = 5.0)
        aa('--rebuild',
           help='forces the search index to be rebuilt',
           default = False, 
           action = 'store_true')
        aa('--nresults',
           help='the number of results to return for each query',
           type = int,
           default = 10)
    
    def _convert_leftovers(self,l):
        # KLUDGE: I need to be able to pass arbitrary kwargs to the search class
        # via the command-line interface. There's probably a better way, but this
        # is my best shot for the moment.
        d = {}
        for i in xrange(len(l)):
            if i % 2:
                # this is a value
                try:
                    # number,bool,list, etc.
                    l[i] = eval(l[i])
                except NameError:
                    # the value is a string. Leave it alone.
                    pass
                d[l[i - 1]] = l[i]
            else:
                # This is a key. Strip leading dashes
                l[i] = l[i].lstrip('-')
        return d
    
    def _parse_known(self):
        return self._parser.parse_known_args()
    
    def _get_search(self,args,searchclass_kwargs):
        _id = 'search/%s' % args.feature    
        module = \
        __import__('zounds.model.framesearch',fromlist = [args.searchclass])
        searchclass = getattr(module,args.searchclass)
        
        if args.rebuild:
            try:
                del searchclass[_id]
            except KeyError:
                # It's ok. No search with that name exists. We'll be rebuilding it
                # anyway.
                pass
    
        try:
            search = searchclass[_id]
        except KeyError:
            search = searchclass(\
                        _id,getattr(self._framemodel,args.feature),**searchclass_kwargs)
            search.build_index()
        
        return search
        
    def setup(self):
        self._add_args()
        args,leftover = self._parse_known()
        if args.maxseconds <= args.minseconds:
            raise ValueError('maxseconds must be greater than minseconds')
        searchclass_kwargs = self._convert_leftovers(leftover)
        return args,self._get_search(args,searchclass_kwargs)
        



class PsychicIter(object):
    '''
    An iterator that can warn, if asked, that the next call to next() will raise
    a StopIteration exception.  This sort of goes against the whole notion of
    iterators, but it's helpful.
    '''
    def __init__(self,iterator):
        self.done = False
        self._iter = iterator
        self._buffer = []
    
    def __iter__(self):
        while True:
            yield self.next()
    
    
    def next(self):
        for i in range(0,2 - len(self._buffer)):
            try:
                self._buffer.append(self._iter.next())
            except StopIteration:
                self.done = True
                break
        
        try:
            return self._buffer.pop(0)
        except IndexError:
            raise StopIteration

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