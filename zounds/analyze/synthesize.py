import numpy as np
from scikits.audiolab import play

class WindowedAudioSynthesizer(object):
    '''
    A very simple synthesizer that assumes windowed, but otherwise raw frames
    of audio.
    '''
    def __init__(self,windowsize,stepsize):
        object.__init__(self)
        self.windowsize = windowsize
        self.stepsize = stepsize
    
    def __call__(self,frames):
        output = np.zeros(self.windowsize + (len(frames) * self.stepsize))
        for i,f in enumerate(frames):
            start = i * self.stepsize
            stop = start + self.windowsize
            output[start : stop] += f
        return output
            
    def play(self,audio):
        output = self(audio)
        return self.playraw(output)
    
    def playraw(self,audio):
        try:
            play(np.tile(audio,(2,1)) * .2)
        except KeyboardInterrupt:
            pass

# TODO: FFT synthesizer
# TODO: DCT synthesizer
# TODO: ConstantQ synthesizer