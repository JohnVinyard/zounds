import numpy as np
import unittest
from preprocess import NoOp, MeanStd,Downsample


class NoOpTests(unittest.TestCase):
    
    def test_noop(self):
        a = np.random.random_sample(100)
        p = NoOp()
        b = p(a)
        self.assertTrue(np.all(a == b))
        
class MeanStdTests(unittest.TestCase):
    
    def test_mean_std(self):
        # KLUDGE: This feels like kind of a stupid test, since the behavior 
        # of MeanStd is so simple.  I'm basically re-implementing its behavior
        # here, and making sure that it does, well, that.
        a = np.random.random_sample((100,100))
        mean = a.mean(0)
        std = (a - mean).std(0)
        p = MeanStd()
        p(a)
        # ensure that mean and std have been set
        self.assertTrue(np.all(mean == p.mean))
        self.assertTrue(np.all(std == p.std))
        c = np.random.random_sample((100,100))
        d = p(c)
        # ensure that mean and std haven't changed
        self.assertTrue(np.all(mean == p.mean))
        self.assertTrue(np.all(std == p.std))
        # ensure that the preprocessor changed c
        self.assertFalse(np.all(c == d))
        # ape the expected behavior of the preprocessor
        expected = c - mean
        expected = expected / std
        self.assertTrue(np.all(expected == d))
        
    
    def test_mean_std_different_shape(self):
        # ensure that data of a different shape cannot be passed in
        # once the preprocessor has been initialized
        a = np.random.random_sample((100,100))
        p = MeanStd()
        p(a)
        
        c = np.random.random_sample((100,11))
        self.assertRaises(ValueError, lambda : p(c))
        
class DownsampleTests(unittest.TestCase):
    
    def test_downsample(self):
        d = Downsample((10,10),2)
        a = np.zeros((100,100))
        ds = d(a)
        self.assertEqual((100,25),ds.shape)