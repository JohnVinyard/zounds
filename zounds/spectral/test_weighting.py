import unittest2
import numpy as np
from weighting import AWeighting
from frequencyscale import LinearScale, FrequencyBand
from tfrepresentation import FrequencyDimension
from zounds.timeseries import Seconds, TimeDimension
from zounds.core import ArrayWithUnits, IdentityDimension


class WeightingTests(unittest2.TestCase):
    def test_cannot_multiply_when_array_does_not_have_expected_dimensions(self):
        td = TimeDimension(Seconds(1), Seconds(1))
        tf = ArrayWithUnits(np.ones((90, 100)), [td, IdentityDimension()])
        weighting = AWeighting()
        self.assertRaises(ValueError, lambda: tf * weighting)

    def test_can_get_weights_from_tf_representation(self):
        td = TimeDimension(Seconds(1), Seconds(1))
        fd = FrequencyDimension(LinearScale(FrequencyBand(20, 22050), 100))
        tf = ArrayWithUnits(np.ones((90, 100)), [td, fd])
        weighting = AWeighting()
        weights = weighting.weights(tf)
        self.assertEqual((100,), weights.shape)

    def test_can_get_weights_from_scale(self):
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        weighting = AWeighting()
        weights = weighting.weights(scale)
        self.assertEqual((100,), weights.shape)
