from __future__ import division
import numpy as np
from resample import Resample
from scikits.audiolab import Sndfile
from nputil import pad


class BadSampleRateException(BaseException):
    '''
    Raised when the sample rate of an audio file
    doesn't match the sample rate of the AudioStream.
    '''
    def __init__(self,expectedrate,actualrate):
        BaseException.__init__(\
            self,
            Exception('Sample rate should have been %i, but was %i' % \
            (expectedrate,actualrate)))
        
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
        
    def _check_samplerate(self,sndfile):
        '''
        Ensure that the file's sample rate matches the sample rate
        used to create this AudioStream instance.
        '''
        if sndfile.samplerate != self.samplerate:
            raise BadSampleRateException(self.samplerate,sndfile.samplerate)
        
    
    def _iter_interchunk(self,interchunk,frames,ws,ss):
        '''
        Iterate over frames that span the gap between chunks
        '''
        #ss = self.stepsize
        #ws = self.windowsize
        if interchunk is not None and len(interchunk):
            for i in xrange(0,self._nsteps-1):
                offset = ss*i
                yield pad(np.concatenate(\
                            [interchunk[offset:],frames[:ss+offset]]),ws)
                
    def _read_frames(self,nframes,sndfile,channels):
        '''
        Read audio samples from a file
        '''
        if channels == 1:
            frames = sndfile.read_frames(nframes)
        elif channels == 2:
            # average the values from the two channels
            frames = sndfile.read_frames(nframes).sum(1) / 2
        return frames
    
    #def _resample(self,frames,snd_sample_rate,channels):
    #    outsamples = np.zeros(self.windowsize,dtype = np.float32)
    #    return resample(frames,outsamples,snd_sample_rate,self.samplerate,channels)
    
        
    def __iter__(self):
        sndfile = Sndfile(self.filename)
        ratio = sndfile.samplerate / self.samplerate
        ws = int(np.ceil(self.windowsize * ratio))
        ss = int(np.ceil(self.stepsize * ratio))
        chunksize = int(np.ceil(self._chunksize * ratio))
        for f in self._iter_frames(sndfile,ws,ss,chunksize):
            if 1 == ratio:
                yield f
            else:
                if None is self._rs:
                    self._rs = Resample(sndfile.samplerate,self.samplerate)
                outsamples = np.zeros(self.windowsize,dtype = np.float32)
                yield self._rs(f,outsamples)
                
        
    def _iter_frames(self,sndfile,ws,ss,chunksize):
        #sndfile = Sndfile(self.filename)
        channels = sndfile.channels
        #self._check_samplerate(sndfile)
        
        #ws = self.windowsize
        #ss = self.stepsize
        #chunksize = self._chunksize
        
        f = 0
        nframes = sndfile.nframes
        firstchunk = True
        interchunk = None
        diff = ws - ss
        while f < nframes:
            # get number of remaining frames
            framesleft = nframes - f 
            # determine if this will be the last chunk we read
            lastchunk = framesleft <= chunksize
            if lastchunk:
                # read all remaining frames
                frames = self._read_frames(framesleft,sndfile,channels)
            else:
                # read the next chunk 
                frames = self._read_frames(chunksize, sndfile, channels)
            
            if lastchunk:
                for ic in self._iter_interchunk(interchunk,frames,ws,ss):
                    yield ic
                
                for i in xrange(0,(len(frames)+1) - ss,ss):
                    yield pad(frames[i:i+ws],ws)
            elif firstchunk:
                for i in xrange(0,len(frames)-diff,ss):
                    yield frames[i:i+ws]
                interchunk = frames[-diff:]
            else:
                for ic in self._iter_interchunk(interchunk,frames,ws,ss):
                    yield ic
                for i in xrange(0,len(frames)-diff,ss):
                    yield frames[i:i+ws]   
                interchunk = frames[-diff:]
                
            firstchunk = False
            f += len(frames)
         
        sndfile.close()
        
        

    
        
        