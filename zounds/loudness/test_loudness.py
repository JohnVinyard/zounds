import unittest2
from loudness import mu_law, inverse_mu_law, inverse_one_hot, instance_scale
import numpy as np


class TestLoudness(unittest2.TestCase):
    def test_can_invert_mu_law(self):
        a = np.random.normal(0, 1, (100, 4))
        adjusted = mu_law(a)
        inverted = inverse_mu_law(adjusted)
        np.testing.assert_allclose(a, inverted)


class TestInverseOneHot(unittest2.TestCase):
    def test_inverse_one_hot_1d_produces_scalar(self):
        arr = np.zeros(10)
        arr[5] = 1
        x = inverse_one_hot(arr)
        self.assertEqual((), x.shape)

    def test_inverse_one_hot_2d_produces_1d_output(self):
        arr = np.eye(10)
        x = inverse_one_hot(arr, axis=-1)
        self.assertEqual((10,), x.shape)
        np.testing.assert_allclose(
            np.linspace(-1, 1, 10, endpoint=False), x, rtol=1e-3)

    def test_inverse_one_hot_3d_produces_2d_output(self):
        arr = np.random.random_sample((3, 128, 256))
        x = inverse_one_hot(arr, axis=1)
        self.assertEqual((3, 256), x.shape)


class TestInstanceScale(unittest2.TestCase):
    def test_handles_zeros(self):
        x = np.random.random_sample((10, 3))
        x[4, :] = 0
        scaled = instance_scale(x, axis=-1)
        maxes = np.ones(10)
        maxes[4] = 0
        np.testing.assert_allclose(maxes, scaled.max(axis=-1))

    def test_handles_negative_numbers_correctly(self):
        x = np.random.normal(0, 1, (10, 300))
        scaled = instance_scale(x, axis=-1)
        np.testing.assert_allclose(1, np.abs(scaled).max(axis=-1))
