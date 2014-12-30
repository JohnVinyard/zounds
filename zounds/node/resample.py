from zounds.flow.extractor import Node

class Resample(Node):
    
    def __init__(self, samplerate = 44100, needs = None):
        super(Resample,self).__init__(needs = needs)
        self._samplerate = samplerate
    
    def _process(self,data):
        raise NotImplemented()