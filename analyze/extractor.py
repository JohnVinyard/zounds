from util import pad

class CircularDependencyException(BaseException):
    '''
    Raised when two extractors directly or indirectly depend
    on one another. 
    '''
    def __init__(self,e1,e2):
        BaseException.__init__(\
            'Circular dependency detected between %s and %s' % (e1,e2))
        
class Extractor(object):
    '''
    An extractor collects input from one or more sources until it
    has enough data to perform some arbitrary calculation.  It then
    makes this calculation available to other extractors that may
    depend on it.
    '''
    
    def __init__(self,needs=None,nframes=1,step=1):
        
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
        
        # a dictionary mapping the extractors on which we depend
        # to the input we'll be collecting from them
        self.input = dict([(src,[]) for src in self.sources])
        
        # a variable into which we'll put our processed data
        self.out = None
        
        # a flag letting other dependent extractors know that
        # we've reached the end of available data
        self.done = False
            
    @property
    def is_root(self):
        '''
        True if this extractor has no sources. True indicates that this
        extractor generates data, rather than drawing it from another extractor
        source.
        '''
        return not self.sources
    
    def _deep_sources(self,accum=None):
        '''
        Gets all direct and indirect dependencies
        '''
        if not accum:
            accum = []
        
        accum.extend(self.sources)
        for src in self.sources:
            src._deep_sources(accum)
        
        return accum
    
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
        '''
        Collect data from the extractors on which we depend
        '''
        
        if all([s.out is not None for s in self.sources]):
            for src in self.sources:
                self.input[src].append(src.out)
        
        if all([s.done for s in self.sources]):
            for src in self.sources:
                if len(self.input[src]):
                    # maybe have a partial input that needs to be padded
                    self.input[src] = pad(self.input[src],self.nframes)
                    
            self.done = True
        

    def _process(self):
        '''
        A hook that derived classes must implement. This is where the feature
        extraction happens
        '''
        raise NotImplemented()
    
    def process(self):
        '''
        Decide if we have enough data to perform feature extraction. If so,
        process the data and get rid of any input data we won't be needing 
        again.
        '''
        full = all([len(v) == self.nframes for v in self.input.values()]) 
        if full:
            # we have enough info to do some processing
            self.out = self._process()
            # remove step size from our inputs
            for src in self.sources:
                self.input[src] = self.input[src][self.step:]
        if not full or self.done:
            self.out = None
            
    def __hash__(self):
        return hash(\
            (self.__class__,tuple(self.sources),self.nframes,self.step))
    
    def __repr__(self):
        return '%s(nframes = %i, step = %i)' % \
            (self.__class__.__name__,self.nframes,self.step)
            
    def __str__(self):
        return self.__repr__()

class SingleInput(Extractor):
    '''
    This class addresses the common case in which an extractor
    will only have a single input. It exposes a property, in_data,
    which is equivalent to self.input[self.sources[0]]
    '''
    def __init__(self,needs,nframes=1,step=1):
        if needs is None:
            raise ValueError('SingleInput extractor cannot be root')
        Extractor.__init__(self,needs=needs,nframes=nframes,step=step)
        
    @property
    def in_data(self):
        '''
        A convenience method allowing easier access to the data from this
        extractor's sole input.
        '''
        return self.input[self.sources[0]]
    
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
        self.chain.sort(self._sort)
        if not self.chain[0].is_root:
            raise RootlessExtractorChainException()
    
    def process(self):
        '''
        A generator that will extract features until the source data runs out
        '''
        while not all([c.done for c in self.chain]):
            for c in self.chain:
                c.collect()
                c.process()
                if c.out is not None:
                    yield c,c.out
                    
                
    def collect(self):
        '''
        Turn the crank until we run out of data
        '''
        bucket = dict([(c,[]) for c in self.chain])
        for k,v in self.process():
            bucket[k].append(v)
        return bucket

    def _sort(self,e1,e2):
        '''
        Sort extractors based on their dependencies. The root extractor
        should always be the first in list, followed by extractors that
        depend on it, etc. 
        '''
        if e1.depends_on(e2) and e2.depends_on(e1):
            raise CircularDependencyException(e1,e2)
        
        if e1.depends_on(e2):
            return 1
        
        if e2.depends_on(e1):
            return -1
        
        return 0
        