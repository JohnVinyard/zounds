from __future__ import division
import numpy as np


# TODO: Write tests
def pad(a,desiredlength):
    '''
    Pad an n-dimensional numpy array with zeros so that it is the
    desired length.  Return it unchanged if it is greater than or
    equal to the desired length
    '''
    
    if len(a) >= desiredlength:
        return a
    
    
    diff = desiredlength - len(a)
    shape = list(a.shape)
    shape[0] = diff
    return np.concatenate([a,np.zeros(shape,dtype=a.dtype)])



def testsignal(hz,seconds=5.,sr=44100.):
    # cycles per sample
    cps = hz / sr
    # total samples
    ts = seconds * sr
    return np.sin(np.arange(0,ts*cps,cps) * (2*np.pi))


from scikits.audiolab import Sndfile,Format
from uuid import uuid4
from time import sleep
import os
if __name__ == '__main__':
    ws = 2048
    sl = 2048 * 2
    sr = 44100
    
    try:
        signal = np.zeros(sl,dtype=np.int16)
        for i,w in enumerate(xrange(0,sl,ws)):
            signal[w:w+ws] = i
            
        fn1 = '%s.wav' % str(uuid4())
        sf1 = Sndfile(fn1,'w',Format(),1,sr)
        sf1.write_frames(signal)
        sf1.sync()
        sf1.close()
        
        sleep(1)
        
        sf1 = Sndfile(fn1,'r',Format(),1,sr)
        sf1_sig = sf1.read_frames(sf1.nframes,dtype=np.int16)
        assert np.all(signal == sf1_sig)
        print 'passed mono'
        
        signal2 = np.tile(signal,(2,1)).T
        fn2 = '%s.wav' % str(uuid4())
        sf2 = Sndfile(fn2,'w',Format(),2,sr)
        sf2.write_frames(signal2)
        sf2.sync()
        sf2.close()
        
        sleep(1)
        
        sf2 = Sndfile(fn2,'r',Format(),2,sr)
        sf2_sig = sf2.read_frames(sf2.nframes,dtype=np.int16)
        print 'ORIGINAL'
        print signal2
        print signal2.shape
        print 'WRONG!!!'
        print sf2_sig
        print sf2_sig.shape
        # Why do I have to do this crazy, convoluted thing to get the same signal?
        # Is there a better, faster numpy way to do this?
        rs = np.concatenate([sf2_sig[:,0],sf2_sig[:,1]]).reshape(signal2.shape)
        print 'RIGHT!!!'
        print rs
        assert np.all(signal2 == rs)
        print 'passed stereo'
        
        assert np.all((rs.sum(1) / 2) == signal)
        print 'passed sum'
    except Exception,e:
        print 'error'
        print e
    

    os.remove(fn1)
    os.remove(fn2)
