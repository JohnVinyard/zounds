from featureflow import BaseModel
from .sliding_window import \
    SlidingWindow, IdentityWindowingFunc, OggVorbisWindowingFunc
from zounds.timeseries import \
    AudioSamples, SR22050, SR44100, SR11025, SR48000, SR96000, SampleRate, \
    Picoseconds, Seconds, Milliseconds, TimeDimension, HalfLapped, TimeSlice
from zounds.util import simple_in_memory_settings
from zounds.basic import resampled
from zounds.synthesize import NoiseSynthesizer
from zounds.core import ArrayWithUnits
from zounds.persistence import ArrayWithUnitsFeature
from zounds.spectral import FrequencyDimension, LinearScale, FrequencyBand
from zounds.synthesize import SilenceSynthesizer
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

    def test_maintains_dtype(self):
        samples = np.random.random_sample(10).astype(np.float32)
        wf = OggVorbisWindowingFunc()
        result = wf * samples
        self.assertEqual(np.float32, result.dtype)


class SlidingWindowTests(unittest2.TestCase):
    def _check(self, samplerate, expected_window_size, expected_step_size):
        samples = AudioSamples(
            np.zeros(5 * samplerate.samples_per_second), samplerate)
        wscheme = samplerate.half_lapped()
        ws, ss = samples._sliding_window_integer_slices(
            TimeSlice(wscheme.duration), TimeSlice(wscheme.frequency))
        self.assertEqual(expected_window_size, ws[0])
        self.assertEqual(expected_step_size, ss[0])

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

    def test_can_apply_sliding_windows_in_succession(self):
        samplerate = SR11025()
        short_window = samplerate * (16, 512)
        long_window = SampleRate(
            frequency=short_window.frequency * 1,
            duration=short_window.frequency * 64)
        rs = resampled(resample_to=samplerate, store_resampled=True)

        samples = AudioSamples.silence(samplerate, Seconds(10))

        @simple_in_memory_settings
        class Sound(rs):
            short_windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=short_window,
                needs=rs.resampled)

            long_windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=long_window,
                needs=short_windowed)

        _id = Sound.process(meta=samples.encode())
        snd = Sound(_id)
        self.assertEqual((512,), snd.short_windowed.shape[1:])
        self.assertEqual((64, 512), snd.long_windowed.shape[1:])

    def test_can_persist_and_retrieve_with_second_long_windowing_scheme(self):
        samplerate = SR22050()
        rs = resampled(resample_to=samplerate)

        window_size = Seconds(1)
        wscheme = SampleRate(window_size, window_size)

        @simple_in_memory_settings
        class Document(rs):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=wscheme,
                needs=rs.resampled,
                store=True)

        synth = NoiseSynthesizer(samplerate)
        audio = synth.synthesize(Milliseconds(5500))

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        self.assertEqual(6, len(doc.windowed))

    def test_has_correct_duration(self):
        samplerate = SR22050()
        rs = resampled(resample_to=samplerate)

        @simple_in_memory_settings
        class Document(rs):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=HalfLapped(),
                needs=rs.resampled,
                store=True)

        synth = NoiseSynthesizer(samplerate)
        audio = synth.synthesize(Milliseconds(5500))

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        orig_seconds = audio.dimensions[0].end / Picoseconds(int(1e12))
        new_seconds = doc.windowed.dimensions[0].end / Picoseconds(int(1e12))
        self.assertAlmostEqual(orig_seconds, new_seconds, delta=0.01)

    def test_can_apply_sliding_window_to_constant_rate_time_series(self):
        arr = ArrayWithUnits(np.zeros(100), [TimeDimension(Seconds(1))])
        sw = SampleRate(Seconds(2), Seconds(2))

        @simple_in_memory_settings
        class Document(BaseModel):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=sw,
                store=True)

        _id = Document.process(windowed=arr)
        result = Document(_id).windowed

        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((50, 2), result.shape)
        self.assertEqual(2, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], TimeDimension)
        self.assertEqual(Seconds(2), result.dimensions[0].frequency)
        self.assertIsInstance(result.dimensions[1], TimeDimension)
        self.assertEqual(Seconds(1), result.dimensions[1].frequency)

    def test_can_apply_sliding_window_to_time_frequency_representation(self):
        band = FrequencyBand(0, 22000)
        scale = LinearScale(band, 100)
        arr = ArrayWithUnits(
            np.zeros((200, 100)),
            [TimeDimension(Seconds(1)), FrequencyDimension(scale)])
        sw = SampleRate(Seconds(2), Seconds(2))

        @simple_in_memory_settings
        class Document(BaseModel):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=sw,
                store=True)

        _id = Document.process(windowed=arr)
        result = Document(_id).windowed

        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((100, 2, 100), result.shape)
        self.assertEqual(3, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], TimeDimension)
        self.assertEqual(Seconds(2), result.dimensions[0].frequency)
        self.assertIsInstance(result.dimensions[1], TimeDimension)
        self.assertEqual(Seconds(1), result.dimensions[1].frequency)
        self.assertIsInstance(result.dimensions[2], FrequencyDimension)

    def test_sliding_window_maintains_dtype(self):
        band = FrequencyBand(0, 22000)
        scale = LinearScale(band, 100)
        arr = ArrayWithUnits(
            np.zeros((200, 100), dtype=np.uint8),
            [TimeDimension(Seconds(1)), FrequencyDimension(scale)])
        sw = SampleRate(Seconds(2), Seconds(2))

        @simple_in_memory_settings
        class Document(BaseModel):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=sw,
                store=True)

        _id = Document.process(windowed=arr)
        result = Document(_id).windowed
        self.assertEqual(np.uint8, result.dtype)
