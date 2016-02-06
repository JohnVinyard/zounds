import unittest2
import numpy as np
from duration import Seconds
from samplerate import SR44100, SampleRate
from audiosamples import AudioSamples


class AudioSamplesTest(unittest2.TestCase):
    def test_raises_if_not_audio_samplerate(self):
        arr = np.zeros(44100 * 2.5)
        one = Seconds(1)
        self.assertRaises(
            TypeError, lambda: AudioSamples(arr, SampleRate(one, one)))

    def test_can_create_instance(self):
        arr = np.zeros(44100 * 2.5)
        instance = AudioSamples(arr, SR44100())
        self.assertIsInstance(instance, AudioSamples)
        length_seconds = instance.end / Seconds(1)
        self.assertAlmostEqual(2.5, length_seconds, places=6)
