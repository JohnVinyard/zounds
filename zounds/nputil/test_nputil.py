import unittest
from toeplitz import toeplitz2dc as toep2d
import numpy as np

class NpUtilTest(unittest.TestCase):
    def test_toeplitz(self):
        a = np.zeros((60,100),dtype = np.float32)
        shape = (30,3)
        t = toep2d(a,shape)
        self.assertEqual((30*97,30*3),t.shape)