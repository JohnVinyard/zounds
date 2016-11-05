from sliding_window import \
    SlidingWindow, IdentityWindowingFunc, OggVorbisWindowingFunc
from zounds.timeseries import \
    AudioSamples, SR22050, SR44100, SR11025, SR48000, SR96000, SampleRate, \
    Seconds, Milliseconds, ConstantRateTimeSeriesFeature
from zounds.util import simple_in_memory_settings
from zounds.basic import resampled
from zounds.synthesize import NoiseSynthesizer
import numpy as np
import unittest2


class WindowingFunctionTests(unittest2.TestCase):
    def test_multiply_lhs(self):
        samples = np.random.random_sample(44)
        wf = IdentityWindowingFunc()
        np.testing.assert_allclose(wf * samples, samples)

    def test_multiply_rhs(self):
        samples = np.random.random_sample(33)
        wf = IdentityWindowingFunc()
        np.testing.assert_allclose(samples * wf, samples)


class OggVorbisWindowingFunctionTests(unittest2.TestCase):
    def test_multilpy_many_frames(self):
        samples = np.random.random_sample((10, 3))
        wf = OggVorbisWindowingFunc()
        result = wf * samples
        self.assertEqual(samples.shape, result.shape)

    def test_multiply_one_frame(self):
        samples = np.random.random_sample((10, 1))
        wf = OggVorbisWindowingFunc()
        result = wf * samples
        self.assertEqual(samples.shape, result.shape)

    def test_multiply_1d(self):
        samples = np.random.random_sample(10)
        wf = OggVorbisWindowingFunc()
        result = wf * samples
        self.assertEqual(samples.shape, result.shape)


class SlidingWindowTests(unittest2.TestCase):
    def _check(self, samplerate, expected_window_size, expected_step_size):
        sw = SlidingWindow(wscheme=samplerate.half_lapped())
        samples = AudioSamples(
                np.zeros(5 * samplerate.samples_per_second), samplerate)
        sw._enqueue(samples, None)
        self.assertEqual(expected_window_size, sw._windowsize)
        self.assertEqual(expected_step_size, sw._stepsize)

    def test_correct_window_and_step_size_at_96000(self):
        self._check(SR96000(), 4096, 2048)

    def test_correct_window_and_step_size_at_48000(self):
        self._check(SR48000(), 2048, 1024)

    def test_correct_window_and_step_size_at_22050(self):
        self._check(SR22050(), 1024, 512)

    def test_correct_window_and_step_size_at_44100(self):
        self._check(SR44100(), 2048, 1024)

    def test_correct_window_and_step_size_at_11025(self):
        self._check(SR11025(), 512, 256)

    def test_can_persist_and_retrieve_with_second_long_windowing_scheme(self):
        samplerate = SR22050()
        rs = resampled(resample_to=samplerate)

        window_size = Seconds(1)
        wscheme = SampleRate(window_size, window_size)

        @simple_in_memory_settings
        class Document(rs):
            windowed = ConstantRateTimeSeriesFeature(
                    SlidingWindow,
                    wscheme=wscheme,
                    needs=rs.resampled,
                    store=True)

        synth = NoiseSynthesizer(samplerate)
        audio = synth.synthesize(Milliseconds(5500))

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        self.assertEqual(6, len(doc.windowed))

    def test_can_apply_sliding_window_to_constant_rate_time_series(self):
        self.fail()

    def test_can_apply_sliding_window_to_time_frequency_representation(self):
        self.fail()
