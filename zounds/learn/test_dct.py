import unittest2
from .dct_transform import DctTransform
import torch
from torch.autograd import Variable
from zounds.spectral import HanningWindowingFunc


class DctTransformTests(unittest2.TestCase):
    def test_can_do_short_time_dct_transform(self):
        t = torch.FloatTensor(3, 1, 512)
        v = Variable(t)
        dct_trasform = DctTransform()
        stdct = dct_trasform.short_time_dct(v, 128, 64, HanningWindowingFunc())
        self.assertEqual((3, 128, 7), tuple(stdct.shape))
