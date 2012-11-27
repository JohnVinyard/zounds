import unittest2
import numpy as np
from zounds.analyze.extractor import ExtractorChain
from spectral import FFT
from zounds.environment import Environment
from zounds.testhelper import RootExtractor

class MockEnvironment(object):
     
    def __init__(self,windowsize):
        self.windowsize = windowsize


class FFTTests(unittest2.TestCase):
    
    def setUp(self):
        self.orig_env = Environment.instance
    
    def tearDown(self):
        Environment.instance = self.orig_env
    
    def test_no_args_correct_dim(self):
        ws = 2048
        Environment.instance = MockEnvironment(ws)
        re = RootExtractor(shape = 100)
        fft = FFT(needs = re)
        self.assertEqual(1024,fft._dim)
    
    def test_no_args_correct_inshape(self):
        ws = 2048
        Environment.instance = MockEnvironment(ws)
        re = RootExtractor(shape = 100)
        fft = FFT(needs = re)
        self.assertEqual((2048,),fft._inshape)
    
    def test_no_args_oned_audio(self):
        ws = 2048
        Environment.instance = MockEnvironment(ws)
        re = RootExtractor(shape = ws)
        fft = FFT(needs = re)
        ec = ExtractorChain([re,fft])
        data = ec.collect()
        fftdata = np.concatenate(data[fft])
        self.assertEqual(1024,fftdata.shape[1])
    
    def test_reshape(self):
        re = RootExtractor(shape = 100)
        fft = FFT(needs = re, inshape = (10,10), axis = 1)
        ec = ExtractorChain([re,fft])
        data = ec.collect()
        # multiple power spectrums are always unravelled, so we should see
        # 10 frames with 50 coefficients each, i.e., each input is of shape
        # (10,10), which is reduced to shape (10,5) by performing an fft over 
        # the first dimension.  Finally, each frame of (10,5) is flattened to 
        # shape (50,)
        fftdata = np.concatenate(data[fft])
        self.assertEqual(50,fftdata.shape[1])
    
    def test_multiframe(self):
        # This test is nearly identical to test_reshape, except that it gathers
        # inputs of size (10,) for 10 frames before processing, instead of 
        # processing inputs of size 100 each frame.
        re = RootExtractor(shape = 10)
        fft = FFT(needs = re, inshape = (10,10), nframes = 10)
        ec = ExtractorChain([re,fft])
        data = ec.collect()
        fftdata = np.concatenate(data[fft])
        self.assertEqual((1,50),fftdata.shape)
        #self.assertTrue(np.all([50 == len(f) for f in data[fft]]))