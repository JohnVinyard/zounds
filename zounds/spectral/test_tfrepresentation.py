from __future__ import division
import numpy as np
import unittest2
from tfrepresentation import TimeFrequencyRepresentation
from frequencyscale import LinearScale, LogScale, FrequencyBand
from weighting import AWeighting
from zounds.timeseries import Seconds


class TimeFrequencyRepresentationTests(unittest2.TestCase):

    def test_from_example(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.ones((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        from_example = TimeFrequencyRepresentation.from_example(
                np.ones((30, 100)), tf)
        self.assertEqual(tf.shape, from_example.shape)
        self.assertEqual(tf.frequency, from_example.frequency)
        self.assertEqual(tf.duration, from_example.duration)
        self.assertEqual(tf.scale, from_example.scale)

    def test_can_multiply_by_frequency_weighting_linear_scale(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.ones((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        result = tf * AWeighting()
        self.assertIsInstance(result, TimeFrequencyRepresentation)
        peak_frequency_band = FrequencyBand(9000, 11000)
        lower_band = FrequencyBand(100, 300)
        peak_slice = np.abs(result[:, peak_frequency_band]).max()
        lower_slice = np.abs(result[:, lower_band]).max()
        self.assertGreater(peak_slice, lower_slice)

    def test_can_multiply_by_frequency_weighting_log_scale(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LogScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.ones((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        result = tf * AWeighting()
        self.assertIsInstance(result, TimeFrequencyRepresentation)
        peak_frequency_band = FrequencyBand(9000, 11000)
        lower_band = FrequencyBand(100, 300)
        peak_slice = np.abs(result[:, peak_frequency_band]).max()
        lower_slice = np.abs(result[:, lower_band]).max()
        self.assertGreater(peak_slice, lower_slice)

    def test_can_multiply_by_array(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.ones((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        result = tf * np.ones(100)
        self.assertIsInstance(result, TimeFrequencyRepresentation)
        np.testing.assert_allclose(tf, result)

    def test_can_convert_to_string(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        s = str(tf)
        self.assertTrue(len(s))

    def test_can_use_single_integer_index(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        indexed = tf[0]
        self.assertEqual((100,), indexed.shape)
        self.assertIsInstance(indexed, TimeFrequencyRepresentation)

    def test_can_use_list_of_integers_as_index(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        indexed = tf[[0, 10, 14]]
        self.assertEqual((3, 100), indexed.shape)
        self.assertIsInstance(indexed, TimeFrequencyRepresentation)

    def test_can_construct_instance(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        self.assertIsInstance(tf, TimeFrequencyRepresentation)

    def test_raises_if_scale_length_does_not_match_frequency_dimension(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 1000)
        self.assertRaises(ValueError, lambda: TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale))

    def test_array_must_be_at_least_2d(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        self.assertRaises(ValueError, lambda: TimeFrequencyRepresentation(
                np.zeros(30),
                frequency=frequency,
                duration=duration,
                scale=scale))

    def test_can_slice_frequency_dimension_with_integer_indices(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        sliced = tf[:, 10: 20]
        self.assertEqual((30, 10), sliced.shape)
        self.assertIsInstance(sliced, TimeFrequencyRepresentation)

    def test_can_slice_frequency_dimensions_with_frequency_band(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        bands = list(scale)
        sliced = tf[:, bands[0]]
        self.assertEqual((30, 1), sliced.shape)
        self.assertIsInstance(sliced, TimeFrequencyRepresentation)

    def test_can_slice_freq_dimension_with_freq_band_spanning_bins(self):
        frequency = Seconds(1)
        duration = Seconds(1)
        scale = LinearScale(FrequencyBand(20, 22050), 100)
        tf = TimeFrequencyRepresentation(
                np.zeros((30, 100)),
                frequency=frequency,
                duration=duration,
                scale=scale)
        bands = list(scale)
        wide_band = FrequencyBand(bands[0].start_hz, bands[9].stop_hz)
        sliced = tf[:, wide_band]
        self.assertEqual((30, 10), sliced.shape)
        self.assertIsInstance(sliced, TimeFrequencyRepresentation)
