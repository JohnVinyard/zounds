import unittest2
from functional import categorical, inverse_categorical
from zounds.core import ArrayWithUnits
from zounds.synthesize import SineSynthesizer
from zounds.timeseries import Seconds, SR11025, TimeSlice
import numpy as np


class CategoricalTests(unittest2.TestCase):
    def test_can_convert_to_categorical_distribution(self):
        samplerate = SR11025()
        synth = SineSynthesizer(samplerate)
        samples = synth.synthesize(Seconds(4), [220, 440, 880])
        _, windowed = samples.sliding_window_with_leftovers(
            TimeSlice(duration=samplerate.frequency * 512),
            TimeSlice(duration=samplerate.frequency * 256))
        c = categorical(windowed, mu=255)
        self.assertEqual(windowed.shape + (255 + 1,), c.shape)
        np.testing.assert_allclose(c.sum(axis=-1), 1)

    def test_can_invert_categorical_distribution(self):
        samplerate = SR11025()
        synth = SineSynthesizer(samplerate)
        samples = synth.synthesize(Seconds(4), [220, 440, 880])
        _, windowed = samples.sliding_window_with_leftovers(
            TimeSlice(duration=samplerate.frequency * 512),
            TimeSlice(duration=samplerate.frequency * 256))
        c = categorical(windowed, mu=255)
        inverted = inverse_categorical(c, mu=255)
        self.assertEqual(windowed.shape, inverted.shape)
        self.assertIsInstance(inverted, ArrayWithUnits)
        self.assertSequenceEqual(windowed.dimensions, inverted.dimensions)

