from __future__ import division
import unittest
from analyze.audiostream import *
from scikits.audiolab import Sndfile,Format
from uuid import uuid4
from os import remove
from math import ceil

# TODO: How do I cleanup wave files for failed tests?
 
class AudioStreamTests(unittest.TestCase):
    
        
    def filename(self):
        return '%s.wav' % str(uuid4())
    
    def make_signal(self,length,winsize,samplerate):
        signal = np.ndarray(length,dtype=np.int16)
        for i,w, in enumerate(xrange(0,length,winsize)):
            signal[w:w+winsize] = i
        return signal
    
    def make_sndfile(self,length,winsize,samplerate):
        signal = self.make_signal(length, winsize, samplerate)
        filename = self.filename() 
        sndfile = Sndfile(filename,'w',Format(),1,samplerate)
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
        
    def test_fileshorterthanchunksize(self):
        step = 1024
        ws = 2048
        length = ws*9
        fn = self.make_sndfile(length, ws, 44100)
        a = AudioStream(fn,44100,ws,step)
        l = [w for w in a]
        [self.assertEqual(ws,len(q)) for q in l]
        el = ceil(length/step)
        self.assertEqual(el,len(l))
        self.assertEqual(l[0][0],0)
        self.assertEqual(l[1][0],0)
        self.assertAlmostEqual(l[1][step],1,4)
        self.remove_sndfile(fn)
        
    def test_fileevenlydivisiblebychunksize(self):
        step = 1024
        ws = 2048
        length = ws * 20
        fn = self.make_sndfile(length, ws, 44100)
        a = AudioStream(fn,44100,ws,step)
        l = [w for w in a]
        [self.assertEqual(ws,len(q)) for q in l]
        el = ceil(length/step)
        self.assertEqual(el,len(l))
        self.assertEqual(l[0][0],0)
        self.assertAlmostEqual(l[3][0],1,4)
        self.assertAlmostEqual(l[3][step],2,4)
        self.remove_sndfile(fn)
        
    def test_filenotevenlydivisiblebychunksize(self):
        step = 1024
        ws = 2048
        length = int(ws * 20.3)
        fn = self.make_sndfile(length, ws, 44100)
        a = AudioStream(fn,44100,ws,step)
        l = [w for w in a]
        el = ceil(length/step)
        [self.assertEqual(ws,len(q)) for q in l]
        self.assertEqual(el,len(l))
        self.remove_sndfile(fn)
        
        
        
    def test_encoding(self):
        #TODO: I've got the encoding (int16) hardcoded in a few places.
        # Fix it!
        self.fail()



