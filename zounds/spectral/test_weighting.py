import unittest2
import numpy as np
from weighting import AWeighting
from frequencyscale import LinearScale, FrequencyBand, GeometricScale, MelScale
from tfrepresentation import FrequencyDimension
from frequencyadaptive import FrequencyAdaptive
from zounds.timeseries import Seconds, TimeDimension, Milliseconds, SR11025
from zounds.core import ArrayWithUnits, IdentityDimension
from functional import fir_filter_bank


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

    def test_can_apply_a_weighting_to_time_frequency_representation(self):
        td = TimeDimension(Seconds(1), Seconds(1))
        fd = FrequencyDimension(LinearScale(FrequencyBand(20, 22050), 100))
        tf = ArrayWithUnits(np.ones((90, 100)), [td, fd])
        weighting = AWeighting()
        result = tf * weighting
        self.assertGreater(result[0, -1], result[0, 0])

    def test_can_apply_a_weighting_to_frequency_adaptive_representation(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.ones((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        weighting = AWeighting()
        result = fa * weighting
        self.assertGreater(
            result[:, scale[-1]].sum(), result[:, scale[0]].sum())

    def test_can_invert_frequency_weighting(self):
        td = TimeDimension(Seconds(1), Seconds(1))
        fd = FrequencyDimension(LinearScale(FrequencyBand(20, 22050), 100))
        tf = ArrayWithUnits(np.random.random_sample((90, 100)), [td, fd])
        weighted = tf * AWeighting()
        inverted = weighted / AWeighting()
        np.testing.assert_allclose(tf, inverted)

    def test_can_invert_frequency_weighting_for_adaptive_representation(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.random.random_sample((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        weighting = AWeighting()
        result = fa * weighting
        inverted = result / AWeighting()
        np.testing.assert_allclose(fa, inverted)

    def test_can_apply_weighting_to_explicit_frequency_dimension(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.ones((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        fa2 = ArrayWithUnits(fa, fa.dimensions)
        weighting = AWeighting()
        result = fa2 * weighting
        self.assertGreater(
            result[:, scale[-1]].sum(), result[:, scale[0]].sum())

    def test_can_invert_weighting_for_explicit_frequency_dimension(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.ones((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        fa2 = ArrayWithUnits(fa, fa.dimensions)
        weighting = AWeighting()
        result = fa2 * weighting
        inverted = result / AWeighting()
        np.testing.assert_allclose(fa, inverted)

    def test_can_apply_weighting_to_filter_bank(self):
        sr = SR11025()
        band = FrequencyBand(20, sr.nyquist)
        scale = MelScale(band, 100)
        bank = fir_filter_bank(scale, 256, sr, np.hanning(25))
        weighted = bank * AWeighting()
        self.assertSequenceEqual(bank.dimensions, weighted.dimensions)

    def test_multiplication_by_weighting_is_commutative(self):
        sr = SR11025()
        band = FrequencyBand(20, sr.nyquist)
        scale = MelScale(band, 100)
        bank = fir_filter_bank(scale, 256, sr, np.hanning(25))
        np.testing.assert_allclose(bank * AWeighting(), AWeighting() * bank)
