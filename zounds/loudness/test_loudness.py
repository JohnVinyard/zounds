import unittest2
from loudness import mu_law, inverse_mu_law
import numpy as np


class TestLoudness(unittest2.TestCase):
    def test_can_invert_mu_law(self):
        a = np.random.normal(0, 1, (100, 4))
        adjusted = mu_law(a)
        inverted = inverse_mu_law(adjusted)
        np.testing.assert_allclose(a, inverted)
