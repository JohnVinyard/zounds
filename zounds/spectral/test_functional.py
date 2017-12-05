import numpy as np
import unittest2
from functional import stft, apply_scale, frequency_decomposition
from zounds.core import ArrayWithUnits
from zounds.synthesize import SilenceSynthesizer, TickSynthesizer
from zounds.timeseries import SR22050, Milliseconds, TimeDimension, TimeSlice
from zounds.spectral import \
    HanningWindowingFunc, FrequencyDimension, LinearScale, GeometricScale, \
    ExplicitFrequencyDimension


class FrequencyDecompositionTests(unittest2.TestCase):
    def test_can_decompose(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(9999))
        wscheme = sr.windowing_scheme(8192, 4096)
        duration = TimeSlice(wscheme.duration)
        frequency = TimeSlice(wscheme.frequency)
        _, windowed = samples.sliding_window_with_leftovers(
            duration, frequency, dopad=True)
        fa = frequency_decomposition(
            windowed, [32, 64, 128, 256, 512, 1024, 2048, 4096])
        self.assertEqual(windowed.dimensions[0], fa.dimensions[0])
        self.assertIsInstance(
            fa.dimensions[1], ExplicitFrequencyDimension)


class STFTTests(unittest2.TestCase):
    def test_has_correct_number_of_bins(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(6666))
        wscheme = sr.windowing_scheme(512, 256)
        tf = stft(samples, wscheme, HanningWindowingFunc())
        self.assertEqual(tf.shape[1], 257)

    def test_has_correct_dimensions(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(6666))
        wscheme = sr.windowing_scheme(512, 256)
        tf = stft(samples, wscheme, HanningWindowingFunc())
        self.assertIsInstance(tf, ArrayWithUnits)
        self.assertEqual(2, len(tf.dimensions))
        self.assertIsInstance(tf.dimensions[0], TimeDimension)
        self.assertEqual(tf.dimensions[0].samplerate, wscheme)
        self.assertIsInstance(tf.dimensions[1], FrequencyDimension)
        self.assertIsInstance(tf.dimensions[1].scale, LinearScale)


class ApplyScaleTests(unittest2.TestCase):
    def test_has_correct_shape(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(9999))
        wscheme = sr.windowing_scheme(256, 128)
        scale = GeometricScale(50, sr.nyquist, 0.4, 32)
        scale.ensure_overlap_ratio()
        tf = stft(samples, wscheme, HanningWindowingFunc())
        geom = apply_scale(tf, scale, window=HanningWindowingFunc())
        self.assertEqual(tf.shape[:-1] + (len(scale),), geom.shape)

    def test_has_correct_dimensions(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(9999))
        wscheme = sr.windowing_scheme(256, 128)
        scale = GeometricScale(50, sr.nyquist, 0.4, 32)
        scale.ensure_overlap_ratio()
        tf = stft(samples, wscheme, HanningWindowingFunc())
        geom = apply_scale(tf, scale, window=HanningWindowingFunc())
        self.assertIsInstance(geom, ArrayWithUnits)
        self.assertEqual(2, len(geom.dimensions))
        self.assertIsInstance(geom.dimensions[0], TimeDimension)
        self.assertEqual(geom.dimensions[0].samplerate, wscheme)
        self.assertIsInstance(geom.dimensions[1], FrequencyDimension)
        self.assertEqual(scale, geom.dimensions[1].scale)

    def test_preserves_time_dimension(self):
        sr = SR22050()
        samples = TickSynthesizer(sr).synthesize(
            Milliseconds(10000), Milliseconds(1000))
        wscheme = sr.windowing_scheme(256, 128)
        scale = GeometricScale(50, sr.nyquist, 0.4, 32)
        scale.ensure_overlap_ratio()
        tf = stft(samples, wscheme, HanningWindowingFunc())
        geom = apply_scale(tf, scale, window=HanningWindowingFunc())

        # get the loudness envelope of each
        tf_envelope = np.abs(tf.real).sum(axis=1)
        geom_envelope = geom.sum(axis=1)

        tf_zeros = np.where(tf_envelope == 0)
        geom_zeros = np.where(geom_envelope == 0)

        np.testing.assert_allclose(tf_zeros, geom_zeros)
