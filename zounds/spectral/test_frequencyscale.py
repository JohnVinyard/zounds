from __future__ import division
import unittest2
from frequencyscale import FrequencyBand, LinearScale, LogScale
from zounds.timeseries import SR44100
import numpy as np


class FrequencyBandTests(unittest2.TestCase):
    def test_can_create_from_center_frequency(self):
        fb = FrequencyBand.from_center(1000, 50)
        self.assertEqual(FrequencyBand(975, 1025), fb)

    def test_does_not_equal_non_frequency_band_class(self):
        fb = FrequencyBand(100, 200)
        self.assertNotEqual(fb, 10)

    def test_cannot_create_with_invalid_interval(self):
        self.assertRaises(ValueError, lambda: FrequencyBand(200, 100))


class FrequencyScaleTests(unittest2.TestCase):

    def test_can_get_all_even_sized_bands(self):
        samplerate = SR44100()
        scale = LinearScale.from_sample_rate(
                samplerate, 44100, always_even=True)
        log_scale = LogScale(FrequencyBand(20, 20000), 64)
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
        scale2 = LogScale(fb2, 100)

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
        np.testing.assert_allclose(linear_freqs, fft_freqs[:-1], rtol=1e-3)

    def test_constant_bandwidth(self):
        scale = LinearScale(FrequencyBand(0, 22050), 1024)
        # taking the second-order differential should result in all zeros
        # if the bandwidths are a constant size
        diff = np.diff(list(scale.center_frequencies), n=2)
        print diff.min(), diff.max()
        np.testing.assert_allclose(diff, np.zeros(len(diff)), atol=1e-11)

    def test_get_slice_on_boundary(self):
        scale = LinearScale(FrequencyBand(0, 1000), 100)
        sl = scale.get_slice(FrequencyBand(500, 700))
        self.assertEqual(slice(49, 70), sl)

    def test_get_slice_between_boundary(self):
        scale = LinearScale(FrequencyBand(0, 1000), 10)
        sl = scale.get_slice(FrequencyBand(495, 705))
        self.assertEqual(slice(4, 7), sl)


class LogScaleTests(unittest2.TestCase):
    def test_variable_bandwidth(self):
        scale = LogScale(FrequencyBand(20, 22050), 100)
        diff = np.diff(list(scale.bandwidths))
        # All differences should be positive, as bandwidth should be
        # monotonically increasing
        self.assertTrue(np.all(diff > 0))
