import numpy as np
from scikits.audiolab import Sndfile
from random import randint,choice
import os


# KLUDGE: This is copied straight from zounds/onsets.py
# it should go in the audio module, not here
def get_signal(file_name):

    '''
    reads the audio signal from a sound file and
    returns a mono signal.  stereo signals are summed
    to mono. mono signals are unaltered.
    '''
    snd = Sndfile(file_name)
    samples = snd.read_frames(snd.nframes)
    if len(samples.shape) == 1:
        return snd.samplerate,samples
    else:
        return snd.samplerate,samples.sum(1)

# the directory where our training data (i.e., audio files) lives
snd_dir = '/home/john/snd/FreeSound/'

# a list of all the files with a .wav extension in snd_dir
files = filter(lambda i : i.endswith('wav'),os.listdir(snd_dir))

def random_filepath():
    '''
    Get the absolute path to a random wav file from the 
    data directory
    '''
    name = choice(files)
    return '%s%s' % (snd_dir,name)
    
def fetch_signal(path):
    '''
    Try to get the signal from the file at path. Every once
    in a while, file is corrupted (or something), and the
    Sndfile class can't read it.
    '''
    try:
        return get_signal(path)
    except IOError:
        # file contains data in an unknown format error
        return -1,None

def random_signal(ps):

    '''
    Get a random signal with a sampling rate of 44100
   '''

    p = random_filepath()
    sr,signal = fetch_signal(p)

    # make sure the sample rate is 44100, and the signal 
    # is at least as large as the patch size.
    while sr != 44100 or len(signal) < ps:
        p = random_filepath()
        sr,signal = fetch_signal(p)

    print p
    return signal

def random_sequence(ps):
    '''
    grab a sequence of sl samples, at random
    '''

    signal = random_signal(ps)
    index = randint(0,len(signal) - ps)
    return signal[index : index + ps]

def random_fft(ps):
    if ps % 2:
        raise ValueError('ps must be even')

    return np.fft.rfft(random_sequence(ps))[1:]


if __name__ == '__main__':
    pass
