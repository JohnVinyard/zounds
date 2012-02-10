from __future__ import division
import numpy as np
import unittest
from analyze.audiostream import \
    AudioStream,BadSampleRateException,BadStepSizeException
from scikits.audiolab import Sndfile,Format
from uuid import uuid4
from os import remove
from math import floor

# TODO: How do I cleanup wave files for failed tests?
 
class AudioStreamTests(unittest.TestCase):
    
        
    def filename(self):
        return '%s.wav' % str(uuid4())
    
    def make_signal(self,length,winsize):
        signal = np.ndarray(int(length),dtype=np.int16)
        for i,w, in enumerate(xrange(0,int(length),winsize)):
            signal[w:w+winsize] = i
        return signal
    
    def make_sndfile(self,length,winsize,samplerate,channels=1):
        signal = self.make_signal(length, winsize)
        filename = self.filename() 
        sndfile = Sndfile(filename,'w',Format(),channels,samplerate)
        if channels == 2:
            signal = np.tile(signal,(2,1)).T
        print signal.dtype
        print signal
        sndfile.write_frames(signal)
        sndfile.close()
        return filename
        
    def remove_sndfile(self,filename):
        remove(filename)
    
    def fail(self):
        self.assertTrue(False)
        
    def test_nonexistentfile(self):
        fn = self.filename()
        self.assertRaises(IOError,AudioStream,fn)
        
    def test_badsamplerate(self):
        fn = self.make_sndfile(44101,2048,22050)
        a = AudioStream(fn)
        self.assertRaises(BadSampleRateException, lambda : a.__iter__().next())
        self.remove_sndfile(fn)
        
    def test_badstepsize(self):
        fn = self.make_sndfile(44101,2048,44100)
        self.assertRaises(BadStepSizeException,AudioStream,fn,44100,2048,1023)
        self.remove_sndfile(fn)
        
    def get_frames(self,
                   length,
                   step = 1024,
                   ws = 2048,
                   samplerate = 44100,
                   channels = 1):
        
        fn = self.make_sndfile(length, ws, samplerate, channels)
        a = AudioStream(fn,samplerate,ws,step)
        l = [w for w in a]
        for q in l:
            print q
        # check that all frames are the proper length
        [self.assertEqual(ws,len(q)) for q in l]
        # get the expected length, in frames
        el = floor(length/step)
        # check that the number of frames is as expected
        self.assertEqual(el,len(l))
        n = (ws/step)
        # do we expect the last frame to be padded?
        padded = (length/step) % 1
        for i,q in enumerate(l):
            qs = set(q)
            print i,qs
            if el > 1 and padded and i >= len(l) - (n-1):
                self.assertEqual(3,len(qs))
            elif not i or not i % n:
                self.assertEqual(1,len(qs))
            else:
                self.assertEqual(2,len(qs))
        self.remove_sndfile(fn)
        
    def test_fileshorterthanwindowsize(self):
        self.get_frames(2000)
        
    def test_fileshorterthanstepsize(self):
        self.get_frames(1000)
        
    def test_fileshorterthanchunksize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk-1))
        
    def test_fileevenlydivisiblebychunksize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2))
        
    def test_filenotevenlydivisiblebychunksize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*1.53))
            
    def test_quarterstepsize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2),step=512)  
    
    def test_quarterstepsize_notevenlydivisible(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*(2.1)),step=512)
        
    def test_stereo(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2),channels=2)
    
    def test_stepsize_equals_windowsize_evenly_divisible(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2),step=2048)
        
    def test_stepsize_equals_windowsize_not_evenly_divisible(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*3.97),step=2048)



