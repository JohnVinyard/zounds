from __future__ import division
import numpy as np
from resample import Resample
from scikits.audiolab import Sndfile
from zounds.nputil import windowed
from zounds.analyze2 import chunksize

def read_frames_mono(sndfile,nframes = None):
    if None is nframes:
        nframes = sndfile.nframes
    
    if sndfile.channels == 1:
        return sndfile.read_frames(nframes)
    elif sndfile.channels == 2:
        # average the values from the two channels
        return sndfile.read_frames(nframes).sum(1) * .5


class AudioStream(object):
    
    def __init__(self,filename,samplerate=44100,windowsize=2048,stepsize=1024):
        self.filename = filename
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.done = False
        
    def _read_frames(self,sndfile):
        
        ratio = sndfile.samplerate / self.samplerate
        if 1 == ratio:
            rs = lambda frames,eoi : frames
        else:
            self._resample = Resample(sndfile.samplerate,self.samplerate)
            rs = self._resample.__call__
            # KLUDGE: The following two lines seem to solve a bug whereby 
            # libsamplerate doesn't generate enough samples the first time
            # src_process is called. We're calling it once here, so the "real"
            # output will come out click-free
            chunksize = int(np.round(self.windowsize * ratio))
            rs(np.zeros(chunksize,dtype = np.float32))
        
        
        def rf(nframes,sndfile,channels,end_of_input = False):
            frames = read_frames_mono(sndfile,nframes)
            r = rs(frames,end_of_input)
            return r
        
        return rf

    def __iter__(self):
        sndfile = Sndfile(self.filename)
        channels = sndfile.channels
        nframes = sndfile.nframes
        framen = 0
        framesleft = nframes - framen
        read_frames = self._read_frames(sndfile)
        ws = self.windowsize
        ss = self.stepsize
        leftover = []
        while framesleft >= chunksize:
            # Read a chunk of samples from the audio file at self.samplerate
            frames = read_frames(chunksize,sndfile,channels,False)
            # Combine any leftover samples from the last chunk with the current chunk
            frames = np.concatenate([leftover,frames])
            # Get a windowed array
            leftover,w = windowed(frames,ws,ss,dopad = False)
            framen += chunksize
            framesleft = nframes - framen
            yield w
            
        # The number of remaining frames is less than the chunksize. Read
        # what's left. 
        frames = read_frames(framesleft,sndfile,channels,True)
        # Combine any leftover frames from the last chunk with the current chunk
        frames = np.concatenate([leftover,frames])
        # Get a windowed array. Pad the input so that all samples are used.
        leftover,w = windowed(frames,ws,ss,dopad = True)
        # we're done with the sound file
        sndfile.close()
        # set a flag indicating that there are no more samples to b read
        self.done = True
        # yield the final chunk
        yield w