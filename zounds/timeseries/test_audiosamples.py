import unittest2
import numpy as np
from duration import Seconds
from samplerate import SR44100, SR11025, SampleRate
from zounds.timeseries import TimeDimension, TimeSlice
from zounds.core import IdentityDimension
from audiosamples import AudioSamples
from zounds.synthesize import SineSynthesizer


class AudioSamplesTest(unittest2.TestCase):

    def test_time_slice_should_return_audio_samples(self):
        silence = AudioSamples.silence(SR11025(), Seconds(10))
        ts = TimeSlice(duration=Seconds(1))
        sliced = silence[ts]
        self.assertIsInstance(sliced, AudioSamples)
        self.assertEqual(int(SR11025()), len(sliced))
        self.assertEqual(SR11025(), sliced.samplerate)

    def test_integer_slice_should_return_scalar(self):
        silence = AudioSamples.silence(SR11025(), Seconds(10))
        sliced = silence[10]
        self.assertIsInstance(sliced, np.float32)

    def test_stereo(self):
        silence = AudioSamples.silence(SR11025(), Seconds(10))
        self.assertEqual(1, silence.channels)
        stereo = silence.stereo
        self.assertEqual(2, stereo.channels)
        self.assertEqual((len(silence), 2), stereo.shape)
        np.testing.assert_allclose(silence, stereo[:, 0])
        np.testing.assert_allclose(silence, stereo[:, 1])

    def test_silence_creates_silence(self):
        silence = AudioSamples.silence(SR11025(), Seconds(10))
        self.assertEqual(0, silence.sum())

    def test_silence_honors_channels(self):
        silence = AudioSamples.silence(SR11025(), Seconds(1), channels=2)
        self.assertEqual((11025, 2), silence.shape)
        self.assertEqual(2, silence.channels)

    def test_silence_like_creates_silence(self):
        silence = AudioSamples.silence(SR11025(), Seconds(1), channels=2)
        silence2 = silence.silence_like(Seconds(2))
        self.assertEqual((22050, 2), silence2.shape)
        self.assertEqual(SR11025(), silence2.samplerate)
        self.assertEqual(2, silence2.channels)

    def test_pad_with_samples_adds_silence_at_end(self):
        synth = SineSynthesizer(SR11025())
        samples = synth.synthesize(Seconds(2))
        padded = samples.pad_with_silence(Seconds(4))
        silence = padded[TimeSlice(start=Seconds(2))]
        self.assertEqual(0, silence.sum())

    def test_raises_if_not_audio_samplerate(self):
        arr = np.zeros(int(44100 * 2.5))
        one = Seconds(1)
        self.assertRaises(
            TypeError, lambda: AudioSamples(arr, SampleRate(one, one)))

    def test_raises_if_array_is_more_than_2d(self):
        arr = np.zeros((int(44100 * 2.5), 2, 2))
        self.assertRaises(
            ValueError, lambda: AudioSamples(arr, SR44100()))

    def test_can_create_instance(self):
        arr = np.zeros(int(44100 * 2.5))
        instance = AudioSamples(arr, SR44100())
        self.assertIsInstance(instance, AudioSamples)
        length_seconds = instance.end / Seconds(1)
        self.assertAlmostEqual(2.5, length_seconds, places=6)

    def test_can_mix_two_instances(self):
        arr = np.ones(int(44100 * 2.5))
        first = AudioSamples(arr, SR44100())
        second = AudioSamples(arr, SR44100())
        mixed = first + second
        self.assertIsInstance(mixed, AudioSamples)
        self.assertEqual(SR44100(), mixed.samplerate)
        np.testing.assert_allclose(mixed, 2)

    def test_cannot_mix_two_instances_with_different_sample_rates(self):
        arr = np.ones(int(44100 * 2.5))
        first = AudioSamples(arr, SR44100())
        second = AudioSamples(arr, SR11025())
        self.assertRaises(ValueError, lambda: first + second)

    def test_can_add_plain_numpy_array(self):
        arr = np.ones(int(44100 * 2.5))
        first = AudioSamples(arr, SR44100())
        second = arr.copy()
        mixed = first + second
        self.assertIsInstance(mixed, AudioSamples)
        self.assertEqual(SR44100(), mixed.samplerate)
        np.testing.assert_allclose(mixed, 2)

    def test_channels_returns_one_for_one_dimensional_array(self):
        arr = np.zeros(int(44100 * 2.5))
        instance = AudioSamples(arr, SR44100())
        self.assertEqual(1, instance.channels)

    def test_channels_returns_two_for_two_dimensional_array(self):
        arr = np.zeros(int(44100 * 2.5))
        arr = np.column_stack((arr, arr))
        instance = AudioSamples(arr, SR44100())
        self.assertEqual(2, instance.channels)

    def test_samplerate_returns_correct_value(self):
        arr = np.zeros(int(44100 * 2.5))
        instance = AudioSamples(arr, SR44100())
        self.assertIsInstance(instance.samplerate, SR44100)

    def test_can_sum_to_mono(self):
        arr = np.zeros(int(44100 * 2.5))
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

    def test_sum_along_time_axis(self):
        arr = np.zeros(int(44100 * 2.5))
        arr = np.column_stack((arr, arr))
        ts = AudioSamples(arr, SR44100())
        result = ts.sum(axis=0)
        self.assertIsInstance(result, np.ndarray)
        self.assertNotIsInstance(result, AudioSamples)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], IdentityDimension)

    def test_sum_along_second_axis(self):
        arr = np.zeros(int(44100 * 2.5))
        arr = np.column_stack((arr, arr))
        ts = AudioSamples(arr, SR44100())
        result = ts.sum(axis=1)
        self.assertIsInstance(result, AudioSamples)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], TimeDimension)
