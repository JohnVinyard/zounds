from __future__ import division
import numpy as np
from resample import Resample
from scikits.audiolab import Sndfile
from nputil import pad

def read_frames_mono(sndfile,nframes = None):
    if None is nframes:
        nframes = sndfile.nframes
    if sndfile.channels == 1:
        return sndfile.read_frames(nframes)
    elif sndfile.channels == 2:
        # average the values from the two channels
        return sndfile.read_frames(nframes).sum(1) / 2

class BadStepSizeException(BaseException):
    '''
    Raised when the window size is not evenly
    divisible by the step size
    '''
    def __init__(self):
        BaseException.__init__(\
            Exception('Window size must be evenly divisible by step size'))
        

class AudioStream(object):
    '''
    Iterates over frames of an audio file. Frames can overlap, as long
    as the window size is evenly divisible by the overlap size.  Low level
    features such as FFT and DCT usually rely on overlapping windows of audio.
    '''
    
    _windows_in_chunk = 10
    
    def __init__(self,filename,samplerate=44100,windowsize=2048,stepsize=1024):
        self.filename = filename
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self._chunksize = self.windowsize * AudioStream._windows_in_chunk
        self._nsteps = self._check_step()
        self._rs = None
        
    def _check_step(self):
        '''
        Ensure that the window size is evenly divisible by the
        step size.
        '''
        nsteps = self.windowsize / self.stepsize 
        if (nsteps) % 1:
            raise BadStepSizeException()
        return int(nsteps)
        
    
    def _iter_interchunk(self,interchunk,frames):
        '''
        Iterate over frames that span the gap between chunks
        '''
        ws = self.windowsize
        ss = self.stepsize
        if interchunk is not None and len(interchunk):
            for i in xrange(0,self._nsteps-1):
                offset = ss*i
                concat = np.concatenate([interchunk[offset:],frames[:ss+offset]])
                yield pad(concat,ws)
    
    
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
            chunksize = int(np.round(self._chunksize * ratio))
            rs(np.zeros(chunksize,dtype = np.float32))
        
        
        def rf(nframes,sndfile,channels,end_of_input = False):
            frames = read_frames_mono(sndfile,nframes)
            r = rs(frames,end_of_input)
            return r
        
        return rf
        
        
    def __iter__(self):
        sndfile = Sndfile(self.filename)
        channels = sndfile.channels
        ratio = sndfile.samplerate / self.samplerate
        
        ws = self.windowsize
        ss = self.stepsize
        chunksize = int(np.round(self._chunksize * ratio))
        
        
        f = 0
        nframes = sndfile.nframes
        firstchunk = True
        interchunk = None
        diff = ws - ss
        
        read_frames = self._read_frames(sndfile)
        actual_frames = 0
        while f < nframes:
            # get number of remaining frames
            framesleft = nframes - f 
            # determine if this will be the last chunk we read
            lastchunk = framesleft <= chunksize
            if lastchunk:
                # read all remaining frames
                # nf is the number of frames before resampling
                nf = framesleft
                frames = read_frames(framesleft,sndfile,channels,True)
            else:
                # read the next chunk
                # nf is the number of frames before resampling
                nf = chunksize 
                frames = read_frames(chunksize, sndfile, channels)
            
            if lastchunk:
                for ic in self._iter_interchunk(interchunk,frames):
                    yield ic
                
                for i in xrange(0,(len(frames)+1) - ss,ss):
                    yield pad(frames[i:i+ws],ws)
            elif firstchunk:
                for i in xrange(0,len(frames)-diff,ss):
                    yield frames[i:i+ws]
                interchunk = frames[-diff:]
            else:
                for ic in self._iter_interchunk(interchunk,frames):
                    yield ic
                for i in xrange(0,len(frames)-diff,ss):
                    yield frames[i:i+ws]   
                interchunk = frames[-diff:]
                
            firstchunk = False
            f += nf
            actual_frames += len(frames)
         
        sndfile.close()
        
        

    
        
        