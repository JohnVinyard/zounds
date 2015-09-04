from flow import Node
import numpy as np

class Slice(Node):
    
    def __init__(self, sl = None, needs = None):
        super(Slice, self).__init__(needs = needs)
        self._sl = sl
    
    def _process(self, data):
        yield data[:, self._sl]

class Sum(Node):
    
    def __init__(self, axis = 0, needs = None):
        super(Sum, self).__init__(needs = needs)
        self._axis = axis
    
    def _process(self, data):
        # TODO: This should be generalized.  Sum will have this same problem
        try:
            data = np.sum(data, axis = self._axis)
        except ValueError:
            print 'ERROR'
            data = data
        if data.shape[0]:
            yield data

class Max(Node):
    
    def __init__(self, axis = 0, needs = None):
        super(Max, self).__init__(needs = needs)
        self._axis = axis
    
    def _process(self, data):
        # TODO: This should be generalized.  Sum will have this same problem
        try:
            data = np.max(data, axis = self._axis)
        except ValueError:
            print 'ERROR'
            data = data
        if data.shape[0]:
            yield data
        
        