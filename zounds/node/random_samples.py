from flow import Node
import numpy as np

class RandomSamples(Node):
    
    def __init__(self, needs = None, probability = None):
        super(RandomSamples,self).__init__(needs = needs)
        self._probability = probability
    
    def _process(self,data):
        n_samples = len(data) // self._probability
        indices = np.random.permutation(len(data))[:n_samples]
        print data[indices].shape
        yield data[indices]