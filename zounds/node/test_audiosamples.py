import unittest2
import numpy as np
from duration import Seconds
from samplerate import SR44100, SR11025, SampleRate
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

    def test_channels_returns_one_for_one_dimensional_array(self):
        arr = np.zeros(44100 * 2.5)
        instance = AudioSamples(arr, SR44100())
        self.assertEqual(1, instance.channels)

    def test_channels_returns_two_for_two_dimensional_array(self):
        arr = np.zeros(44100 * 2.5)
        arr = np.column_stack((arr, arr))
        instance = AudioSamples(arr, SR44100())
        self.assertEqual(2, instance.channels)

    def test_samplerate_returns_correct_value(self):
        arr = np.zeros(44100 * 2.5)
        instance = AudioSamples(arr, SR44100())
        self.assertIsInstance(instance.samplerate, SR44100)

    def test_can_sum_to_mono(self):
        arr = np.zeros(44100 * 2.5)
        arr = np.column_stack((arr, arr))
        instance = AudioSamples(arr, SR44100())
        mono = instance.mono
        self.assertEqual(1, mono.channels)
        self.assertIsInstance(mono.samplerate, SR44100)

    def test_class_concat_returns_audio_samples(self):
        s1 = AudioSamples(np.zeros(44100 * 2), SR44100())
        s2 = AudioSamples(np.zeros(44100), SR44100())
        s3 = AudioSamples.concat([s1, s2])
        self.assertIsInstance(s3, AudioSamples)
        self.assertEqual(44100 * 3, len(s3))

    def test_instance_concat_returns_audio_samples(self):
        s1 = AudioSamples(np.zeros(44100 * 2), SR44100())
        s2 = AudioSamples(np.zeros(44100), SR44100())
        s3 = s1.concat([s1, s2])
        self.assertIsInstance(s3, AudioSamples)
        self.assertEqual(44100 * 3, len(s3))

    def test_concat_raises_for_different_sample_rates(self):
        s1 = AudioSamples(np.zeros(44100 * 2), SR44100())
        s2 = AudioSamples(np.zeros(44100), SR11025())
        self.assertRaises(ValueError, lambda: AudioSamples.concat([s1, s2]))



