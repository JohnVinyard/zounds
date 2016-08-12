import unittest2
import numpy as np
from weighting import AWeighting
from tfrepresentation import TimeFrequencyRepresentation
from frequencyscale import LinearScale, FrequencyBand
from zounds.timeseries import Seconds


class WeightingTests(unittest2.TestCase):
    def test_can_get_weights_from_tf_representation(self):
        t = Seconds(1)
        tf = TimeFrequencyRepresentation(
            np.ones((90, 100)),
            frequency=t,
            duration=t,
            scale=LinearScale(FrequencyBand(20, 22050), 100))
        weighting = AWeighting()
        weights = weighting.weights(tf)
        self.assertEqual((100,), weights.shape)

    def test_can_get_weights_from_scale(self):
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        weighting = AWeighting()
        weights = weighting.weights(scale)
        self.assertEqual((100,), weights.shape)
