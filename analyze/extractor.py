from util import pad
'''
FFT()
DCT()
Bark(needs=FFT)
BFCC(needs=FFT)
Onset(needs=Bark,10)
Tempo(needs=Bark,10)
RBM1(needs=Bark,5)
CONVRBM(needs=RBM1,15)

'''

class CircularDependencyException(BaseException):
    def __init__(self,e1,e2):
        BaseException.__init__(\
            'Circular dependency detected between %s and %s' % (e1,e2))
        
class Extractor(object):
    
    def __init__(self,needs,nframes=1,step=1):
        self.source = needs
        self.nframes = nframes
        self.step = step
        self.input = []
        self.out = None
        self.done = False
        
    def __cmp__(self,other):
        '''
        If we need each other, raise an exception!
        If I need other, I am greater than
        If other needs me, I am less than
        If we have no relationship, 0
        '''
        if self.source == other and other.source == self:
            raise CircularDependencyException(self,other)
        
        if self.source == other:
            return 1
        
        if other.source == self:
            return -1
        
        return 0
        
    def __str__(self):
        return self.__class__.__name__
        
    def collect(self):
        if self.source.done:
            self.input = pad(self.input,self.nframes)
            self.done = True
            return
         
        if self.source.out:
            self.input.append(self.source.out)
        
    def _process(self):
        raise NotImplemented()
    
    def process(self):
        if len(self.input) == self.nframes:
            # we have enough info to do some processing
            self.out = self._process()
            self.input = self.input[self.step:]
        else:
            self.out = None
        