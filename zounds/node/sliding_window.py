from zounds.flow.extractor import Node

# TODO: Steal AudioStream tests and use them here
class SlidingWindow(Node):
    
    def __init__(self, windowsize = None, stepsize = None, needs = None):
        super(SlidingWindow,self).__init__(needs = needs)
        self._windowsize = windowsize
        self._stepsize = stepsize
        self._cache = None
    
    def _enqueue(self,data,pusher):
        # first, if cache is None, initialize cache to the data being passed
        # otherwise, concatenate data with cache
        pass

    def _dequeue(self):
        # return windowed cache value, and set cache = leftovers
        pass
    
    def __finalize(self):
        # if cache is not None or empty, then return windowed cache with
        # dopad = true
        pass
    
    def _process(self,data):
        # I don't need to implement this, it should just work
        raise NotImplemented()