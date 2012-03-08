from audiostream import AudioStream
from extractor import Extractor,SingleInput
import numpy as np

# TODO: Implement Pitch, BFCC, Centroid, Flatness, Bark, Tempo, Chroma, 
# Onset, Autocorrelation, DCT

class RawAudio(Extractor):
    
    def __init__(self,filename,samplerate,windowsize,stepsize):
        Extractor.__init__(self)
        self.stream = AudioStream(\
                            filename,samplerate,windowsize,stepsize).__iter__()
        self.key = 'audio'
        
    def _process(self):
        try:
            return self.stream.next()
        except StopIteration:
            self.out = None
            self.done = True
    
    def __hash__(self):
        return hash(\
                    (self.__class__.__name__,
                     self.filename,
                     self.samplerate,
                     self.windowsize,
                     self.step))

class FFT(SingleInput):
    
    def __init__(self,needs=None,key=None):
        SingleInput.__init__(self,needs=needs,nframes=1,step=1,key=key)
        
    def _process(self):
        return np.abs(np.fft.rfft(self.in_data[0]))[1:]
    

class Loudness(SingleInput):
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        SingleInput.__init__(self,needs=needs,nframes=nframes,step=step,key=key)
        
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
    
    