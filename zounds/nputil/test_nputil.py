import unittest
import numpy as np
from npx import windowed,sliding_window,downsample

class DownsampleTest(unittest.TestCase):
    
    def test_downsample_single_example_1D(self):
        a = np.ones(10)
        ds = downsample(a,2)
        self.assertEqual((5,),ds.shape)
    
    def test_downsample_single_example_2D(self):
        a = np.ones((11,11))
        ds = downsample(a,2)
        self.assertEqual((5,5),ds.shape)
    
    def test_downsample_multi_example_1D(self):
        a = np.ones((31,10))
        ds = downsample(a,(2,))
        self.assertEqual((31,5),ds.shape)
    
    def test_downsample_multi_example_2D(self):
        a = np.ones((31,11,11))
        ds = downsample(a,(2,2))
        self.assertEqual((31,5,5),ds.shape)
        
class SlidingWindowTest(unittest.TestCase):
    
    def test_mismatched_dims_ws(self):
        a = np.zeros(10)
        self.assertRaises(ValueError, lambda : sliding_window(a,(1,2)))
    
    def test_mismatched_dims_ss(self):
        a = np.zeros(10)
        self.assertRaises(ValueError, lambda : sliding_window(a,3,(1,2)))
    
    def test_windowsize_too_large_1D(self):
        a = np.zeros(10)
        self.assertRaises(ValueError, lambda : sliding_window(a,11))
    
    def test_windowsize_too_large_2D(self):
        a = np.zeros((10,10))
        self.assertRaises(ValueError, lambda : sliding_window(a,(3,11)))
    
    def test_1D_no_step_specified(self):
        a = np.arange(10)
        b = sliding_window(a,3)
        self.assertEqual((3,3),b.shape)
        self.assertTrue(np.all(b.ravel() == a[:9]))
    
    def test_1D_with_step(self):
        a = np.arange(10)
        b = sliding_window(a,3,1)
        self.assertEqual((8,3),b.shape)
    
    def test_1D_flat_nonflat_equivalent(self):
        a = np.zeros(10)
        bflat = sliding_window(a,3)
        bnonflat = sliding_window(a,3,flatten = False)
        self.assertEqual(bflat.shape,bnonflat.shape)
        self.assertTrue(np.all(bflat == bnonflat))
    
    def test_2D_no_step_specified(self):
        a = np.arange(64).reshape((8,8))
        b = sliding_window(a,(4,4))
        self.assertEqual((4,4,4),b.shape)
    
    def test_2D_with_step(self):
        a = np.zeros((8,8))
        b = sliding_window(a,(4,4),(1,1))
        self.assertEqual((25,4,4),b.shape)
    
    def test_2D_nonflat_no_step_specified(self):
        a = np.arange(64).reshape((8,8))
        b = sliding_window(a,(4,4),flatten = False)
        self.assertEqual((2,2,4,4),b.shape)
    
    def test_2D_nonflat_with_step(self):
        a = np.zeros((8,8))
        b = sliding_window(a,(4,4),(1,1),flatten = False)
        self.assertEqual((5,5,4,4),b.shape)
        
    
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
        self.assertEqual(1,l.shape[0])
        self.assertEqual((9,2),w.shape)
    
    def test_windowsize_two_stepsize_one_pad(self):
        a = np.arange(10)
        l,w = windowed(a,2,1,True)
        self.assertEqual(0,l.shape[0])
        self.assertEqual((10,2),w.shape)
    
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
        self.assertEqual(2,l.shape[0])
        self.assertEqual((4,3),w.shape)
    
    def test_windowsize_three_stepsize_two_pad(self):
        a = np.arange(10)
        l,w = windowed(a,3,2,dopad = True)
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
        self.assertEqual(2,l.shape[0])
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
        
        