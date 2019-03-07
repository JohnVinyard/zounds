
import unittest2
from .frequencyscale import \
    FrequencyBand, LinearScale, ExplicitScale, GeometricScale, Hertz, Hz
from zounds.timeseries import SR44100
import numpy as np


class HertzTests(unittest2.TestCase):
    def test_can_negate_hertz(self):
        hz = -Hertz(10)
        self.assertIsInstance(hz, Hertz)

    def test_can_add_hertz(self):
        hz = Hertz(20) + Hertz(30)
        self.assertEqual(Hertz(50), hz)

    def test_can_subtract_hertz(self):
        hz = Hz(100) - Hz(20)
        self.assertEqual(Hz(80), hz)


class FrequencyBandTests(unittest2.TestCase):
    def test_can_create_from_center_frequency(self):
        fb = FrequencyBand.from_center(1000, 50)
        self.assertEqual(FrequencyBand(975, 1025), fb)

    def test_does_not_equal_non_frequency_band_class(self):
        fb = FrequencyBand(100, 200)
        self.assertNotEqual(fb, 10)

    def test_cannot_create_with_invalid_interval(self):
        self.assertRaises(ValueError, lambda: FrequencyBand(200, 100))

    def test_identical_frequency_bands_have_same_hash_value(self):
        fb1 = FrequencyBand.from_center(1000, 50)
        fb2 = FrequencyBand.from_center(1000, 50)
        self.assertEqual(hash(fb1), hash(fb2))

    def test_can_intersect(self):
        fb1 = FrequencyBand(0, 100)
        fb2 = FrequencyBand(50, 150)
        intersection = fb1.intersect(fb2)
        self.assertEqual(FrequencyBand(50, 100), intersection)

    def test_error_raised_when_no_intersection(self):
        fb1 = FrequencyBand(0, 100)
        fb2 = FrequencyBand(200, 500)
        self.assertRaises(ValueError, lambda: fb1.intersect(fb2))

    def test_intersection_ratio(self):
        fb1 = FrequencyBand(0, 100)
        fb2 = FrequencyBand(50, 150)
        ratio = fb1.intersection_ratio(fb2)
        self.assertEqual(0.5, ratio)

    def test_audible_range_lower_bound(self):
        band = FrequencyBand.audible_range(SR44100())
        self.assertEqual(20, band.start_hz)

    def test_audible_range_upper_bound(self):
        sr = SR44100()
        band = FrequencyBand.audible_range(sr)
        self.assertEqual(int(sr) // 2, band.stop_hz)


class FrequencyScaleTests(unittest2.TestCase):
    def test_get_slice_converts_frequency_band_to_integer_based_slice(self):
        scale = LinearScale(FrequencyBand(0, 100), 10)
        slce = scale.get_slice(FrequencyBand(0, 20))
        self.assertEqual(slice(0, 2), slce)

    def test_get_slice_converts_hz_based_slice_to_integer_based_slice(self):
        scale = LinearScale(FrequencyBand(0, 100), 10)
        slce = scale.get_slice(slice(Hertz(0), Hertz(20)))
        self.assertEqual(slice(0, 2), slce)

    def test_get_slice_returns_integer_based_slice_unaltered(self):
        scale = LinearScale(FrequencyBand(0, 100), 10)
        slce = scale.get_slice(slice(0, 20))
        self.assertEqual(slice(0, 20), slce)

    def test_can_get_all_even_sized_bands(self):
        samplerate = SR44100()
        scale = LinearScale.from_sample_rate(
            samplerate, 44100, always_even=True)
        log_scale = GeometricScale(20, 20000, 0.01, 64)
        slices = [scale.get_slice(band) for band in log_scale]
        sizes = [s.stop - s.start for s in slices]
        self.assertTrue(
            not any([s % 2 for s in sizes]),
            'All slice sizes should be even but were {sizes}'
                .format(**locals()))

    def test_can_get_single_band(self):
        fb1 = FrequencyBand(20, 20000)
        scale1 = LinearScale(fb1, 100)
        fb2 = scale1[10]
        self.assertIsInstance(fb2, FrequencyBand)

    def test_can_get_sub_scale(self):
        fb1 = FrequencyBand(20, 20000)
        scale1 = LinearScale(fb1, 100)
        scale2 = scale1[10:20]
        self.assertIsInstance(scale2, LinearScale)
        self.assertEqual(10, scale2.n_bands)

    def test_equals(self):
        fb1 = FrequencyBand(20, 20000)
        scale1 = LinearScale(fb1, 100)

        fb2 = FrequencyBand(20, 20000)
        scale2 = LinearScale(fb2, 100)

        self.assertEqual(scale1, scale2)

    def test_not_equal_when_scale_differs(self):
        fb1 = FrequencyBand(20, 20000)
        scale1 = LinearScale(fb1, 100)

        fb2 = FrequencyBand(20, 20000)
        scale2 = GeometricScale(20, 20000, 0.01, 100)

        self.assertNotEqual(scale1, scale2)

    def test_not_equal_when_span_differs(self):
        fb1 = FrequencyBand(20, 20000)
        scale1 = LinearScale(fb1, 100)

        fb2 = FrequencyBand(20, 10000)
        scale2 = LinearScale(fb2, 100)

        self.assertNotEqual(scale1, scale2)

    def test_not_equal_when_bands_differ(self):
        fb1 = FrequencyBand(20, 20000)
        scale1 = LinearScale(fb1, 100)

        fb2 = FrequencyBand(20, 20000)
        scale2 = LinearScale(fb2, 50)

        self.assertNotEqual(scale1, scale2)


class LinearScaleTests(unittest2.TestCase):
    def test_matches_fftfreq(self):
        samplerate = SR44100()
        n_bands = 2048
        fft_freqs = np.fft.rfftfreq(n_bands, 1 / int(samplerate))
        bands = LinearScale.from_sample_rate(samplerate, n_bands // 2)
        linear_freqs = np.array([b.start_hz for b in bands])
        np.testing.assert_allclose(linear_freqs, fft_freqs[:-1])

    def test_constant_bandwidth(self):
        scale = LinearScale(FrequencyBand(0, 22050), 1024)
        # taking the second-order differential should result in all zeros
        # if the bandwidths are a constant size
        diff = np.diff(list(scale.center_frequencies), n=2)
        np.testing.assert_allclose(diff, np.zeros(len(diff)), atol=1e-11)

    def test_get_slice_on_boundary(self):
        scale = LinearScale(FrequencyBand(0, 1000), 100)
        sl = scale.get_slice(FrequencyBand(500, 700))
        self.assertEqual(slice(49, 70), sl)

    def test_get_slice_between_boundary(self):
        scale = LinearScale(FrequencyBand(0, 1000), 10)
        sl = scale.get_slice(FrequencyBand(495, 705))
        self.assertEqual(slice(4, 8), sl)

    def test_start_hz(self):
        scale = LinearScale(FrequencyBand(100, 500), 4)
        start_hz = [b.start_hz for b in scale]
        self.assertEqual([100, 200, 300, 400], start_hz)


class GeometricScaleTests(unittest2.TestCase):
    def test_slicing_geometric_scale_returns_explicit_scale(self):
        scale = GeometricScale(
            start_center_hz=20,
            stop_center_hz=5000,
            bandwidth_ratio=0.05,
            n_bands=100)
        sliced = scale[FrequencyBand(100, 1000)]
        self.assertIsInstance(sliced, ExplicitScale)

    def test_ensure_minimal_inersection_ratio_no_overlap(self):
        scale = GeometricScale(
            start_center_hz=300,
            stop_center_hz=3030,
            bandwidth_ratio=0.001,
            n_bands=300)

        self.assertRaises(
            AssertionError,
            lambda: scale.ensure_overlap_ratio(0.5))

    def test_ensure_minimal_inersection_ratio_insufficient_overlap(self):
        scale = GeometricScale(
            start_center_hz=300,
            stop_center_hz=3030,
            bandwidth_ratio=0.01,
            n_bands=300)

        self.assertRaises(
            AssertionError,
            lambda: scale.ensure_overlap_ratio(0.5))

    def test_ensure_minimal_intersection_ratio(self):
        scale = GeometricScale(
            start_center_hz=300,
            stop_center_hz=3030,
            bandwidth_ratio=0.017,
            n_bands=300)

        try:
            scale.ensure_overlap_ratio(0.5)
        except AssertionError:
            self.fail('AssertionError was raised')


class ExplicitScaleTests(unittest2.TestCase):
    def test_can_construct_explicit_scale_from_scale(self):
        linear_scale = LinearScale(FrequencyBand(100, 1000), n_bands=50)
        explicit_scale = ExplicitScale(linear_scale)
        self.assertSequenceEqual(linear_scale.bands, explicit_scale.bands)

    def test_can_construct_explicit_scale_from_iterable_of_bands(self):
        linear_scale = LinearScale(FrequencyBand(100, 1000), n_bands=50)
        explicit_scale = ExplicitScale(linear_scale.bands)
        self.assertSequenceEqual(linear_scale.bands, explicit_scale.bands)

    def test_equals(self):
        scale1 = ExplicitScale(
            LinearScale(FrequencyBand(100, 1000), n_bands=50))
        scale2 = ExplicitScale(
            LinearScale(FrequencyBand(100, 1000), n_bands=50))
        self.assertEqual(scale1, scale2)
