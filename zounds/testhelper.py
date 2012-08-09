import numpy as np
from uuid import uuid4
from scikits.audiolab import Sndfile,Format

def filename():
    return '%s.wav' % str(uuid4())

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