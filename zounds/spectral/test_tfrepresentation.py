from __future__ import division
import numpy as np
import unittest2
from tfrepresentation import TimeFrequencyRepresentation
from frequencyscale import LinearScale, FrequencyBand
from zounds.timeseries import Seconds


class TimeFrequencyRepresentationTests(unittest2.TestCase):

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
