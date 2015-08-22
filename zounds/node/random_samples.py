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
    
    # TODO: What happens if we have filled up all the sample slots and we run
    # out of data?
    def _enqueue(self, data, pusher):
        print data.shape
        if self._r is None:
            shape = (self._nsamples,) + data.shape[1:]
            self._r = np.zeros(shape, dtype = data.dtype)
        print self._r.shape
        
        diff = 0
        if self._index < self._nsamples:
            diff = self._nsamples - self._index
            available = len(data[:diff])
            self._r[self._index : self._index + available] = data[:diff]
            self._index += available
        
        remaining = len(data[diff:])
        if not remaining: return
        indices = np.random.random_integers(0,self._index,size = remaining)
        indices = indices[indices < self._nsamples]
        self._r[indices] = data[diff:][range(len(indices))]
        self._index += remaining
    
    def _dequeue(self):
        if not self._finalized: return
        
        if self._index <= self._nsamples:
            arr = self._r[:self._index]
            np.random.shuffle(arr)
            return arr
        
        return self._r