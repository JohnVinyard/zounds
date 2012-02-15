from audiostream import AudioStream
from extractor import Extractor,SingleInput
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

class FFT(SingleInput):
    
    def __init__(self,needs):
        SingleInput.__init__(self,needs=needs,nframes=1,step=1)
        
    def _process(self):
        return np.abs(np.fft.rfft(self.in_data[0]))[1:]
    

class Loudness(SingleInput):
    
    def __init__(self,needs,nframes=1,step=1):
        SingleInput.__init__(self,needs=needs,nframes=nframes,step=step)
        
    def _process(self):
        return np.sum(self.in_data)
    
    
    

from extractor import ExtractorChain
from matplotlib import pyplot as plt
import optparse


if __name__ == '__main__':
    parser = optparse.OptionParser()
    aa = parser.add_option
    aa('--filename', help='path to a wav, aiff, or flac file', dest='filename')
    options, args = parser.parse_args()
    
    raw = RawAudio(options.filename)
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
    
    