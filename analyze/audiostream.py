from __future__ import division
import numpy as np
from scikits.audiolab import Sndfile
from util import pad
from os.path import exists

# TODO: Write tests
# TODO: Write docs
# TODO: Handle stereo files
# TODO: Handle encodings other than int16
# TODO: Handle other sample rates by re-sampling


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
    
    def __init__(self,filename,samplerate=44100,windowsize=2048,stepsize=1024):
        '''
        '''
        self.filename = filename
        if not exists(self.filename):
            raise IOError('%s does not exist' % filename)
        
        self.samplerate = samplerate
        self.windowsize = windowsize
        self.stepsize = stepsize
        self._chunksize = self.windowsize*10
        self._nsteps = self._checkstep() 
        
    def _checkstep(self):
        nsteps = self.windowsize / self.stepsize 
        if (nsteps) % 1:
            raise BadStepSizeException()
        return int(nsteps)
        
    def _checksamplerate(self,sndfile):
        if sndfile.samplerate != self.samplerate:
            raise BadSampleRateException(self.samplerate,sndfile.samplerate)
        
    def __iter__(self):
        sndfile = Sndfile(self.filename)
        self._checksamplerate(sndfile)
        
        f = 0
        nframes = sndfile.nframes
        firstchunk = True
        interchunk = None
        
        while f < nframes:
            framesleft = nframes - f 
            lastchunk = framesleft <= self._chunksize
            if lastchunk:
                frames = sndfile.read_frames(framesleft,dtype=np.int16)
            else:
                frames = sndfile.read_frames(self._chunksize,dtype=np.int16) 
            
            
            if lastchunk:
                # interchunk
                if interchunk is not None and len(interchunk):
                    for i in xrange(1,self._nsteps):
                        offset = self.stepsize*i
                        yield np.concatenate(\
                            [interchunk[offset:],frames[:self.windowsize-offset]])
                    
                for i in xrange(0,len(frames),self.stepsize):
                    yield pad(frames[i:i+self.windowsize],self.windowsize)
            elif firstchunk:
                
                for i in xrange(0,len(frames)-self.stepsize,self.stepsize):
                    yield frames[i:i+self.windowsize]
                interchunk = frames[-self.windowsize:]
            else:
                # interchunk
                for i in xrange(1,self._nsteps):
                    offset = self.stepsize*i
                    yield np.concatenate(\
                        [interchunk[offset:],frames[:self.windowsize-offset]])
                
                for i in xrange(0,len(frames)-self.stepsize,self.stepsize):
                    yield frames[i:i+self.windowsize]   
                interchunk = frames[-self.windowsize:]
                
            firstchunk = False
            f += len(frames)
         
        sndfile.close()
        

    
        
        