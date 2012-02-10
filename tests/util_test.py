import numpy as np
import unittest
from util import pad


class UtilTests(unittest.TestCase):
    
    def test_pad_onedim_desired(self):
        a = np.array([1,2,3])
        b = pad(a,3)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
        
    def test_pad_onedim_longer(self):
        a = np.array([1,2,3,4])
        b = pad(a,3)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
        
    def test_pad_onedim_shorter(self):
        a = np.array([1,2,3])
        b = pad(a,4)
        self.assertEqual(4,len(b))
        self.assertEqual(0,b[-1])
        
    def test_pad_twodim_desired(self):
        a = np.random.random_sample((10,10))
        b = pad(a,10)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
    
    def test_pad_twodim_longer(self):
        a = np.random.random_sample((12,10))
        b = pad(a,10)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
        
    def test_pad_twodim_shorter(self):
        a = np.random.random_sample((10,10))
        b = pad(a,13)
        self.assertEqual(13,b.shape[0])
        self.assertTrue(np.allclose(a,b[:10]))
        self.assertTrue(np.all(b[10:] == 0))
        
    def test_pad_list(self):
        l = [1,2,3]
        b = pad(l,4)
        self.assertEqual(4,len(b))
        self.assertEqual(0,b[-1])
        