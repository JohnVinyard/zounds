from flow import Node
import numpy as np

class ReservoirSampler(Node):
    '''
    Use reservoir sampling (http://en.wikipedia.org/wiki/Reservoir_sampling) to
    draw a fixed-size set of random samples from a stream of unknown size.
    
    This is useful when the samples can fit into memory, but the stream cannot.
    '''
    def __init__(self, needs = None, nsamples = None):
        super(ReservoirSampler,self).__init__(needs = needs)
        self._nsamples = nsamples
        self._r = None
        self._index = 0
    
    def _enqueue(self, data, pusher):
        if self._r is None:
            shape = (self._nsamples,) + data.shape[1:]
            self._r = np.zeros(shape, dtype = data.dtype)
        
        diff = 0
        if self._index < self._nsamples:
            diff = self._nsamples - self._index
            available = len(data[:diff])
            self._r[self._index : self._index + available] = data[:diff]
            self._index += available
        
        remaining = len(data[diff:])
        indices = np.random.random_integers(0,self._index,size = remaining)
        indices = indices[indices < self._nsamples]
        self._r[indices] = data[diff:][range(len(indices))]
        self._index += remaining
    
    def _dequeue(self):
        if self._finalized: return self._r

class ChunkedStreamingSampler(Node):
    # first, fill the pool in random order by choosing (random pool, random index)
    #
    # then write each new sample to a random (pool,index)
    #
    # then, once a chunk in the pool has received some number of writes maybe 
    # (50 - 75% of its size), deem it well mixed, and yield it.  Reset its 
    # write count
    #
    # keep doing this until we run out of samples 
    def __init__(self, needs = None, chunksize = None, poolsize = None):
        super(ChunkedStreamingSampler,self).__init__(needs = needs)
        self._chunksize = chunksize
        self._poolsize = poolsize
        self._pool = None
        self._mix = None
    
    def _enqueue(self, data, pusher):
        pass
    
    def _dequeue(self):
        # TODO: return the indices of any well-mixed chunks
        pass
    
    def _process(self,data):
        # TODO: yield each well-mixed chunk (data will simply be a set of indices)
        pass