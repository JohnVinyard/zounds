from util import pad
'''
Audio()
FFT()
DCT()
Bark(needs=FFT)
BFCC(needs=FFT)
Onset(needs=[Bark,Audio],10)
Tempo(needs=Bark,10)
RBM1(needs=Bark,5)
CONVRBM(needs=RBM1,15)


'''
# TODO: Write tests

# TODO: Write Docs

# TODO: ExtractorChains need to know about _id, so they can create
# useful queue items to be placed into a data store

class CircularDependencyException(BaseException):
    def __init__(self,e1,e2):
        BaseException.__init__(\
            'Circular dependency detected between %s and %s' % (e1,e2))
        
class Extractor(object):
    '''
    TODO: Write docs
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
        return not self.sources
    
    def _deep_sources(self,accum=None):
        if not accum:
            accum = []
        
        accum.extend(self.sources)
        for src in self.sources:
            src._deep_sources(accum)
        
        return accum
    
    def depends_on(self,e):
        return e in self._deep_sources()
    
    def _root_collect(self):
        pass
    
    def collect(self):
        
        if all([s.done for s in self.sources]):
            for src in self.sources:
                if len(self.input[src]):
                    # we have a partial input that needs to be padded
                    self.input[src] = pad(self.input[src],self.nframes)
            self.done = True
            return
         
        if all([s.out is not None for s in self.sources]):
            for src in self.sources:
                self.input[src].append(src.out)
        
    def _process(self):
        raise NotImplemented()
    
    def process(self):
        if all([len(v) == self.nframes for v in self.input.values()]):
            # we have enough info to do some processing
            self.out = self._process()
            # remove step size from our inputs
            for src in self.sources:
                self.input[src] = self.input[src][self.step:]
        else:
            self.out = None
            
    def __hash__(self):
        return hash((self.__class__,self.nframes,self.step))
    
    def __repr__(self):
        return '%s(nframes = %i, step = %i)' % \
            (self.__class__.__name__,self.nframes,self.step)
            
    def __str__(self):
        return self.__repr__()
    
    
class RootlessExtractorChainException(BaseException):
    def __init__(self):
        BaseException.__init__(self,\
            'An extractor chain must contain at least root extractor.')
            
class ExtractorChain(object):
    '''
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
        while not all([c.done for c in self.chain]):
            for c in self.chain:
                c.collect()
                c.process()
                if c.out is not None:
                    yield c,c.out
                
    def collect(self):
        bucket = dict([(c,[]) for c in self.chain])
        for k,v in self.process():
            bucket[k].append(v)
        return bucket

    def _sort(self,e1,e2):
        '''
        '''
        if e1.depends_on(e2) and e2.depends_on(e1):
            raise CircularDependencyException(e1,e2)
        
        if e1.depends_on(e2):
            return 1
        
        if e2.depends_on(e1):
            return -1
        
        return 0
        