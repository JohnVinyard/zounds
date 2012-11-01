import os
import shutil
from uuid import uuid4
import numpy as np
from scikits.audiolab import Sndfile,Format
from zounds.analyze.extractor import Extractor

class RootExtractor(Extractor):
    
    def __init__(self,shape=1,totalframes=10,chunksize = 5,key = None):
        self.shape = shape
        Extractor.__init__(self,key = key)
        self.framesleft = totalframes
        self.chunksize = chunksize
    
    
    def dim(self,env):
        return self.shape
    
    @property
    def dtype(self):
        return np.int32
    
    def _process(self):
        out = np.ones(self.chunksize) if self.shape == 1 \
                else np.ones((self.chunksize,self.shape))
        self.framesleft -= self.chunksize
        if self.framesleft <= 0:
            self.done = True
            self.out = None
        return out

class SumExtractor(Extractor):
    
    def __init__(self,needs,nframes,step,key = None):
        Extractor.__init__(self,needs,nframes,step, key = key)
        
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.int32
    
    def _sum(self,a):
        return a.sum(axis = 1) if len(a.shape) > 1 else a
    
    def _prepare(self,l):
        # ensure that each sum is at least two dimesnions, so it can be combined
        # with others along the first axis
        for i in xrange(len(l)):
            if len(l[i].shape) == 1:
                l[i] = l[i].reshape((l[i].size,1))
        return l
        
    def _process(self):
        # Sum the sums of all inputs, example-wise. This should result in a 1D array
        src_sums = [self._sum(v) for v in self.input.values()]
        src_sums = self._prepare(src_sums)
        # Combine all the features for each example
        combined = np.concatenate(src_sums,axis = 1)
        # take the sum, example-wise
        return self._sum(combined)

def filename(extension = '.wav'):
    return '%s%s' % (str(uuid4()),extension)

def make_signal(length,winsize):
    '''
    Create a signal which has each successive non-overlapping frame of size
    winsize set to the corresponding frame number. This makes tests a bit easier
    to write, since it's easy to figure out what value(s) should be in a given
    frame.
    '''
    signal = np.ndarray(int(length))
    for i,w, in enumerate(xrange(0,int(length),winsize)):
        signal[w:w+winsize] = i
    return signal


def make_sndfile(length,winsize,samplerate,channels = 1):
    signal = make_signal(length, winsize)
    fn = filename() 
    sndfile = Sndfile(fn,'w',Format(),channels,samplerate)
    if channels == 2:
        signal = np.tile(signal,(2,1)).T
    sndfile.write_frames(signal)
    sndfile.close()
    return fn

def remove(path):
    '''
    Attempt to remove a file or directory. Fail silently if the file doesn't exist,
    or we don't have proper permissions.
    '''
    if os.path.isfile(path):
        try:
            os.remove(path)
        except IOError:
            # the file doesn't exist, or we don't have permission to 
            # delete it
            pass
        return
    
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            # the directory doesn't exist, or we don't have permission to
            # delete it
            pass
        
        