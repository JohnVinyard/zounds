from __future__ import division
import numpy as np
from scikits.audiolab import Sndfile
from util import pad
from os.path import exists


# TODO: Write docs





class BadSampleRateException(BaseException):
    '''
    '''
    def __init__(self,expectedrate,actualrate):
        BaseException.__init__(\
            self,
            Exception('Sample rate should have been %i, but was %i' % \
            (expectedrate,actualrate)))
        
class BadStepSizeException(BaseException):
    '''
    '''
    def __init__(self):
        BaseException.__init__(\
            Exception('Window size must be evenly divisible by step size'))
        

class AudioStream(object):
    '''
    '''
    
    _windows_in_chunk = 10
    
    def __init__(self,filename,samplerate=44100,windowsize=2048,stepsize=1024):
        '''
        '''
        self.filename = filename
        if not exists(self.filename):
            raise IOError('%s does not exist' % filename)
        
        self.samplerate = samplerate
        self.encoding = np.int16
        self.windowsize = windowsize
        self.stepsize = stepsize
        self._chunksize = self.windowsize * AudioStream._windows_in_chunk
        self._nsteps = self._check_step() 
        
    def _check_step(self):
        nsteps = self.windowsize / self.stepsize 
        if (nsteps) % 1:
            raise BadStepSizeException()
        return int(nsteps)
        
    def _check_samplerate(self,sndfile):
        if sndfile.samplerate != self.samplerate:
            raise BadSampleRateException(self.samplerate,sndfile.samplerate)
        
    
    def _iter_interchunk(self,interchunk,frames):
        ss = self.stepsize
        ws = self.windowsize
        if interchunk is not None and len(interchunk):
            for i in xrange(0,self._nsteps-1):
                offset = ss*i
                yield pad(np.concatenate(\
                            [interchunk[offset:],frames[:ss+offset]]),ws)
                
    def _read_frames(self,nframes,sndfile,channels):
        if channels == 1:
            frames = sndfile.read_frames(nframes,dtype=self.encoding)
        elif channels == 2:
            frames = sndfile.read_frames(nframes,dtype=self.encoding)
            # KLUDGE: This is ugly. Is there a quicker, easier, prettier
            # way to get the results I want?
            frames = \
                np.concatenate([frames[:,0],frames[:,1]])\
                .reshape((len(frames),2))
            frames = frames.sum(1) / 2
        return frames
        
        
    def __iter__(self):
        sndfile = Sndfile(self.filename)
        print sndfile
        channels = sndfile.channels
        self._check_samplerate(sndfile)
        
        ws = self.windowsize
        ss = self.stepsize
        
        f = 0
        nframes = sndfile.nframes
        firstchunk = True
        interchunk = None
        diff = ws - ss
        while f < nframes:
            # get number of remaining frames
            framesleft = nframes - f 
            # determine if this will be the last chunk we read
            lastchunk = framesleft <= self._chunksize
            if lastchunk:
                # read all remaining frames
                frames = self._read_frames(framesleft,sndfile,channels)
            else:
                # read the next chunk 
                frames = self._read_frames(self._chunksize, sndfile, channels)
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
            f += len(frames)
         
        sndfile.close()
        

    
        
        