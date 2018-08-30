import unittest2
import torch
from util import batchwise_unit_norm
import numpy as np


class BatchwiseUnitNormTests(unittest2.TestCase):
    def test_all_elements_have_unit_norm(self):
        t = torch.FloatTensor(100, 5).normal_(0, 1)
        t = batchwise_unit_norm(t).data.numpy()
        norms = np.linalg.norm(t, axis=1)
        np.testing.assert_allclose(norms, 1, rtol=1e-6)

    def test_maintains_correct_shape_2d(self):
        t = torch.FloatTensor(100, 5).normal_(0, 1)
        t = batchwise_unit_norm(t).data.numpy()
        self.assertEqual((100, 5), t.shape)

    def test_maintains_correct_shape_3d(self):
        t = torch.FloatTensor(100, 5, 3).normal_(0, 1)
        t = batchwise_unit_norm(t).data.numpy()
        self.assertEqual((100, 5, 3), t.shape)

    def test_does_not_introduce_nans(self):
        t = torch.FloatTensor(100, 5, 3).zero_()
        t = batchwise_unit_norm(t).data.numpy()
        self.assertFalse(np.any(np.isnan(t)))
