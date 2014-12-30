from zounds.flow.extractor import Node

class SlidingWindow(Node):
    
    def __init__(self, windowsize = None, stepsize = None, needs = None):
        super(SlidingWindow,self).__init__(needs = needs)
        self._windowsize = windowsize
        self._stepsize = stepsize
    
    def _process(self,data):
        raise NotImplemented()