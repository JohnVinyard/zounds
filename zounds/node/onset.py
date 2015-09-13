from flow import Node
from timeseries import ConstantRateTimeSeries
import numpy as np

class Flux(Node):
    
    def __init__(self, needs = None):
        super(Flux, self).__init__(needs = needs)
        self._memory = None
    
    def _process(self, data):
        
        if self._memory is None:
            # prepend the first vector, so that the initial flux value is zero
            d = np.vstack([data[0], data])
        else:
            # prepend the last vector from the previous batch
            d = np.vstack([self._memory, data])
        
        self._memory = data[-1]
        
        # Take the difference, keeping only positive changes 
        # (the magnitude increased)
        diff = np.diff(d, axis = 0)
        diff[diff < 0] = 0
        
        # take the l1 norm of each frame
        yield ConstantRateTimeSeries(\
              diff.sum(axis = 1),
              data.frequency,
              data.duration)