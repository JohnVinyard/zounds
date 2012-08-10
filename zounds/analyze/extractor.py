import numpy as np
from abc import ABCMeta,abstractproperty,abstractmethod
from zounds.nputil import pad
from zounds.util import recurse,sort_by_lineage
from zounds.nputil import windowed
from zounds.environment import Environment

class CircularDependencyException(BaseException):
    '''
    Raised when two extractors directly or indirectly depend
    on one another. 
    '''
    def __init__(self,e1,e2):
        BaseException.__init__(\
            'Circular dependency detected between %s and %s' % (e1,e2))


class Extractor(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, needs = None, nframes = 1, step = 1, key = None):
        
        self.set_sources(needs = needs)
        
        # the number of frames needed from all sources to
        # do work
        if nframes < 1:
            raise ValueError('nframes must be greater than or equal to 1')
        self.nframes = nframes
        
        # step size, i.e., how much do we slide nframes
        # each time we compute new features
        if step < 1:
            raise ValueError('step must be greater than or equal to 1')
        self.step = step
        
        # a variable into which we'll put our processed data
        self.out = None
        
        # a flag letting other dependent extractors know that
        # we've reached the end of available data
        self.done = False
        
        # a unique identifier for this extractor
        self.key = key
        
        # True only if this extractor has no stopping condition. Users of an
        # extractor chain containing infinite extractors must be aware of this
        # fact to avoid creating artifically long lists of features
        self.finite = True
    
    
    @abstractmethod
    def dim(self,env):
        '''
        A tuple representing the dimensions of a single frame of output from 
        this extractor
        '''
        pass
    
    @abstractproperty
    def dtype(self):
        '''
        The numpy dtype that will be output by this extractor
        '''
        pass
    
    def set_sources(self,needs = None):
        # a list of other extractors needed by this one
        # to do work
        if not needs:
            # this is a root extractor; it relies on no one.
            self.sources = []
            self.collect = self._root_collect
        elif not isinstance(needs,list):
            self.sources = [needs]
        else:
            self.sources = needs
            
        # a dictionary mapping the extractors on which we depend
        # to the input we'll be collecting from them
        self.input = dict([(src,[]) for src in self.sources])
        # a dictionary mapping the extractors on which we depend
        # to the leftover portions of the inputs we'll be collecting from
        self.leftover = dict([(src,None) for src in self.sources])

        
    @property
    def is_root(self):
        '''
        True if this extractor has no sources. True indicates that this
        extractor generates data, rather than drawing it from another extractor
        source.
        '''
        return not self.sources
    
    @recurse
    def _deep_sources(self):
        return self.sources
    
    
    def nframes_abs(self, nframes = None):
        '''
        Since nframes is expressed in terms of the extractors upon which this 
        one depends, the nframes property doesn't necessarily return an answer 
        in absolute frames.  
        
        This method does its best to answer the question "how many absolute 
        frames must be processed before this extractor can compute its feature"?
        '''
        if nframes is None:
            nframes = [self.nframes]
        
        if self.sources:
            # KLUDGE: We're choosing the max number of frames from all extractors
            # on which we depend. Hopefully, there will either be only a single
            # requisite extractor, OR, the multiple sources will have the same
            # nframes values.  I can't really imagine a situation in which 
            # neither would be true at the moment, but such a situation would
            # break this method.
            nframes.append(max([e.nframes for e in self.sources]))
            for s in self.sources:
                s.nframes_abs(nframes = nframes)
                
        # the product of the "path" of nframes values leading back to the root
        # extractor should tell us how many absolute frames this extractor needs    
        return np.product(nframes)
             
    def step_abs(self,step = None):
        if step is None:
            step = [self.step]
        
        if self.sources:
            step.append(max([e.step for e in self.sources]))
            for s in self.sources:
                s.step_abs(step = step)
        
        return np.product(step)
            
    def depends_on(self,e):
        '''
        Returns true if this extractor directly or indirectly depends on e
        '''
        return e in self._deep_sources()
    
    def _root_collect(self):
        '''
        This noop method is called by root extractors. They generate data, so
        there's nothing to collect.
        '''
        pass
    
    def collect(self):
        alldone = all([s.done for s in self.sources])
        if all([s.out is not None for s in self.sources]):
            for src in self.sources:
                indata = src.out
                if self.leftover[src] is None:
                    self.leftover[src] = np.zeros((0,) + indata.shape[1:])
                indata = np.concatenate([self.leftover[src],indata])
                leftover,data = windowed(\
                                indata,self.nframes,self.step,dopad = alldone)
                self.input[src].append(data)
                self.leftover[src] = leftover
        
        self.done = alldone
    
    # KLUDGE: I'm assuming that chunksize will always be larger than 
    # the largest nframes value. 
    def process(self):
        # Ensure that there's enough data to perform processing
        full = all([len(self.input[src]) for src in self.sources])
        if not full:
            self.out = None
            return

        for src in self.sources:            
            data = np.array(self.input[src][0])    
            self.input[src] = data
    
        self.out = self._process()
        for src in self.sources:
            self.input[src] = []
    
    @abstractmethod
    def _process(self):
        '''
        A hook that derived classes must implement. This is where the feature
        extraction happens
        '''
        raise NotImplemented()
            
    def __hash__(self):
        '''
        Hash this instance based on class name, sources, frame and step size.
        IMPORTANT!!: Derived classes with more/fewer/different parameters
        should always override this method
        '''
        return hash(\
            (self.__class__.__name__,
             self.key,
             frozenset(self.sources),
             self.nframes,
             self.step))
    
    def __eq__(self,other):
        return self.__hash__() == other.__hash__()
    
    def __ne__(self,other):
        return not self.__eq__(other)
    
    def __repr__(self):
        return '%s(key = %s, nframes = %i, step = %i)' % \
            (self.__class__.__name__,self.key,self.nframes,self.step)
            
    def __str__(self):
        return self.__repr__()



class SingleInput(Extractor):
    '''
    This class addresses the common case in which an extractor
    will only have a single input. It exposes a property, in_data,
    which is equivalent to self.input[self.sources[0]]
    '''
    def __init__(self,needs,nframes=1,step=1,key=None):
        if needs is None:
            raise ValueError('SingleInput extractor cannot be root')
        Extractor.__init__(self,needs=needs,nframes=nframes,step=step,key=key)
        self._input = None
        
    @property
    def in_data(self):
        '''
        A convenience method allowing easier access to the data from this
        extractor's sole input.
        '''
        return self.input[self.sources[0]]
    
    def _process(self):
        raise NotImplemented()
    
    def dim(self,env):
        raise NotImplemented()
    
    @property
    def dtype(self):
        raise NotImplemented()

class RootlessExtractorChainException(BaseException):
    '''
    Raised when an extractor chain is made up entirely of consumers; there's
    no producer at the head of the line to fetch or generate data.
    '''
    def __init__(self):
        BaseException.__init__(self,\
            'An extractor chain must contain at least root extractor.')
            
class ExtractorChain(object):
    '''
    A pipeline of feature extractors. An example might be:
    
    raw audio -> FFT -> MFCC
    '''
    
    def __init__(self,extractors):
        if not extractors:
            raise ValueError('An empty extractor chain is not valid')
        elif not isinstance(extractors,list):
            self.chain = [extractors]
        else:
            self.chain = extractors
        self.chain.sort(sort_by_lineage(Extractor._deep_sources))
        if not self.chain[0].is_root:
            raise RootlessExtractorChainException()
        
    def __len__(self):
        '''
        The number of extractors in this chain
        '''
        return self.chain.__len__()

    def __contains__(self,extractor):
        return extractor in self.chain
    
    def __iter__(self):
        '''
        Iterate over the extractors
        '''
        return self.chain.__iter__()
    
    def bucket(self):
        return dict([(c.key if c.key else c,[]) for c in self.chain])
    
    def process(self):
        '''
        A generator that will extract features until the source data runs out
        '''
        while not all([c.done for c in self.chain]):
            for c in self.chain:
                c.collect()
                c.process()
                if c.out is not None:
                    yield c.key if c.key else c,c.out
            
                    
                
    def collect(self,nframes = None):
        '''
        Turn the crank until we run out of data
        '''
        rootkey = self.chain[0].key
        i = 0
        bucket = self.bucket()  
        for k,v in self.process():
            bucket[k].append(v)
            if k == rootkey:
                i+=1
                if nframes is not None and i > nframes:
                    break
        return bucket
    
    
    def  __getitem__(self,key):
        
        if isinstance(key,int):
            return self.chain[key]
        
        if isinstance(key,str):
            r = filter(lambda e : e.key == key, self.chain)
            if not r:
                raise KeyError(key)
            return r[0]
        
        raise ValueError('key must be a string or int')
    
    
    def prune(self,*keys_to_keep):
        '''
        Return a new extractor chain that includes only extractors necessary
        to compute the supplied keys
        '''
        l = set()
        for k in keys_to_keep:
            e = self[k]
            l.add(e)
            l.update(e._deep_sources())
        return ExtractorChain(list(l))
        