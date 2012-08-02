import unittest
from toeplitz import toeplitz2dc as toep2d
import numpy as np
from npx import windowed

class NpUtilTest(unittest.TestCase):
    
    def test_toeplitz(self):
        a = np.zeros((60,100),dtype = np.float32)
        shape = (30,3)
        t = toep2d(a,shape)
        self.assertEqual((30*97,30*3),t.shape)
    

class WindowedTest(unittest.TestCase):
    
    def test_windowsize_ltone(self):
        a = np.arange(10)
        self.assertRaises(ValueError, lambda : windowed(a,0,1))
    
    def test_stepsize_ltone(self):
        a = np.arange(10)
        self.assertRaises(ValueError, lambda : windowed(a,1,0))
    
    def test_no_windowing(self):
        a = np.arange(10)
        l,w = windowed(a,1,1)
        self.assertTrue(a is w)
        self.assertEqual(0,l.shape[0])
    
    def test_drop_samples(self):
        a = np.arange(10)
        l,w = windowed(a,1,2)
        self.assertEqual(5,w.shape[0])
        self.assertEqual(0,l.shape[0])
    
    def test_windowsize_two_stepsize_one_cut(self):
        a = np.arange(10)
        l,w = windowed(a,2,1)
        print w
        self.assertEqual(0,l.shape[0])
        self.assertEqual((9,2),w.shape)
    
    def test_windowsize_two_stepsize_one_pad(self):
        a = np.arange(10)
        l,w = windowed(a,2,1,True)
        print w
        self.assertEqual(0,l.shape[0])
        self.assertEqual((9,2),w.shape)
    
    def test_windowsize_two_stepsize_two_cut(self):
        a = np.arange(10)
        l,w = windowed(a,2,2)
        self.assertEqual(0,l.shape[0])
        self.assertEqual((5,2),w.shape)
    
    def test_windowsize_two_stepsize_two_pad(self):
        a = np.arange(10)
        l,w = windowed(a,2,2,True)
        self.assertEqual(0,l.shape[0])
        self.assertEqual((5,2),w.shape)
    
    def test_windowsize_three_stepsize_two_cut(self):
        a = np.arange(10)
        l,w = windowed(a,3,2)
        self.assertEqual(1,l.shape[0])
        self.assertEqual((4,3),w.shape)
    
    def test_windowsize_three_stepsize_two_pad(self):
        a = np.arange(10)
        l,w = windowed(a,3,2,dopad = True)
        print w
        self.assertEqual(0,l.shape[0])
        self.assertEqual((5,3),w.shape)
        self.assertTrue(np.all([0,0] == w[-1,-1]))
    
    def test_windowsize_three_stepsize_three_cut(self):
        a = np.arange(10)
        l,w = windowed(a,3,3)
        self.assertEqual(1,l.shape[0])
        self.assertEqual((3,3),w.shape)
    
    def test_windowsize_three_stepsize_three_pad(self):
        a = np.arange(10)
        l,w = windowed(a,3,3,dopad = True)
        print w
        self.assertEqual(0,l.shape[0])
        self.assertEqual((4,3),w.shape)
    
    def test_windowsize_gt_length_cut(self):
        a = np.arange(5)
        l,w = windowed(a,6,1)
        self.assertEqual(5,l.shape[0])
        self.assertEqual(0,w.shape[0])
    
    def test_windowsize_gt_length_pad(self):
        a = np.arange(5)
        l,w = windowed(a,6,1,dopad = True)
        self.assertEqual(0,l.shape[0])
        self.assertEqual((1,6),w.shape)
    
    def test_twod_cut(self):
        a = np.arange(20).reshape((10,2))
        l,w = windowed(a,3,2)
        self.assertEqual(1,l.shape[0])
        self.assertEqual((4,3,2),w.shape)
    
    def test_twod_pad(self):
        a = np.arange(20).reshape((10,2))
        l,w = windowed(a,3,2,dopad = True)
        self.assertEqual(0,l.shape[0])
        self.assertEqual((5,3,2),w.shape)
    
    def test_no_stepsize_specified(self):
        a = np.arange(10)
        l,w = windowed(a,2)
        self.assertEqual(0,l.shape[0])
        self.assertEqual((5,2),w.shape)
        
        