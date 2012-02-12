from audiostream import AudioStream
from extractor import Extractor
import numpy as np


class RawAudio(Extractor):
    
    def __init__(self,filename):
        Extractor.__init__(self)
        self.stream = AudioStream(filename).__iter__()
        
    def _process(self):
        try:
            return self.stream.next()
        except StopIteration:
            self.out = None
            self.done = True

class FFT(Extractor):
    
    def __init__(self,needs,nframes=1,step=1):
        Extractor.__init__(self,needs=needs,nframes=nframes,step=step)
        
    def _process(self):
        # TODO: These inputs are the wrong shape.
        inp = self.input.items()
        return np.abs(np.fft.rfft(inp[0][1][0])[1:])
    

class Loudness(Extractor):
    
    def __init__(self,needs,nframes=1,step=1):
        Extractor.__init__(self,needs=needs,nframes=nframes,step=step)
        
    def _process(self):
        inp = self.input.items()
        # TODO: These inputs are the wrong shape.
        return np.sum(inp[0][1][0])
    
    
    

from extractor import ExtractorChain
from matplotlib import pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa('--filename',help='path to a wav, aiff, or flac file')
    args = parser.parse_args()
    
    raw = RawAudio(args.filename)
    fft = FFT(needs=raw)
    loud = Loudness(needs=fft,nframes=2,step=2)
    d = ExtractorChain([loud,raw,fft]).collect()
    
    fftdata = np.array(d[fft])
    fftdata = fftdata[:,:200]
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(np.rot90(np.log(fftdata)))
    plt.subplot(2,1,2)
    plt.plot(d[loud])
    plt.show()
    plt.savefig('features.png')
    plt.clf()
    
    