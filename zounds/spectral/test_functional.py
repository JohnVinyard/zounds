import numpy as np
import unittest2
from functional import \
    fft, stft, apply_scale, frequency_decomposition, phase_shift
from zounds.core import ArrayWithUnits
from zounds.synthesize import \
    SilenceSynthesizer, TickSynthesizer, SineSynthesizer, FFTSynthesizer
from zounds.timeseries import SR22050, SR11025, Seconds, Milliseconds, \
    TimeDimension, TimeSlice
from zounds.spectral import \
    HanningWindowingFunc, FrequencyDimension, LinearScale, GeometricScale, \
    ExplicitFrequencyDimension, FrequencyBand


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


class FFTTests(unittest2.TestCase):
    def test_can_pad_for_better_frequency_resolution(self):
        samples = SilenceSynthesizer(SR22050()).synthesize(Milliseconds(2500))
        windowsize = TimeSlice(duration=Milliseconds(200))
        stepsize = TimeSlice(duration=Milliseconds(100))
        _, windowed = samples.sliding_window_with_leftovers(
            windowsize=windowsize, stepsize=stepsize, dopad=True)
        coeffs = fft(windowed, padding_samples=1024)
        self.assertIsInstance(coeffs, ArrayWithUnits)
        self.assertEqual(2, len(coeffs.dimensions))
        self.assertEqual(windowed.dimensions[0], coeffs.dimensions[0])
        self.assertIsInstance(coeffs.dimensions[1], FrequencyDimension)
        expected_size = ((windowed.shape[-1] + 1024) // 2) + 1
        self.assertEqual(expected_size, coeffs.shape[-1])

    def test_can_take_fft_of_1d_signal(self):
        samples = SilenceSynthesizer(SR22050()).synthesize(Milliseconds(2500))
        coeffs = fft(samples)
        self.assertIsInstance(coeffs, ArrayWithUnits)
        self.assertEqual(1, len(coeffs.dimensions))
        self.assertIsInstance(coeffs.dimensions[0], FrequencyDimension)

    def test_can_take_fft_of_2d_stacked_signal(self):
        samples = SilenceSynthesizer(SR22050()).synthesize(Milliseconds(2500))
        windowsize = TimeSlice(duration=Milliseconds(200))
        stepsize = TimeSlice(duration=Milliseconds(100))
        _, windowed = samples.sliding_window_with_leftovers(
            windowsize=windowsize, stepsize=stepsize, dopad=True)
        coeffs = fft(windowed)
        self.assertIsInstance(coeffs, ArrayWithUnits)
        self.assertEqual(2, len(coeffs.dimensions))
        self.assertEqual(windowed.dimensions[0], coeffs.dimensions[0])
        self.assertIsInstance(coeffs.dimensions[1], FrequencyDimension)


class PhaseShiftTests(unittest2.TestCase):
    def _mean_squared_error(self, x, y):
        l = min(len(x), len(y))
        return ((x[:l] - y[:l]) ** 2).mean()

    def test_1d_phase_shift_returns_correct_size(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Milliseconds(5500), [220, 440, 880])
        coeffs = fft(samples)
        shifted = phase_shift(
            coeffs=coeffs,
            samplerate=samplerate,
            time_shift=Milliseconds(5500),
            frequency_band=FrequencyBand(50, 5000))
        self.assertEqual(coeffs.shape, shifted.shape)

    def test_can_phase_shift_1d_signal(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Milliseconds(5000), [220, 440, 880])
        coeffs = fft(samples)
        shifted = phase_shift(coeffs, samplerate, Milliseconds(10))
        new_samples = np.fft.irfft(shifted, norm='ortho')
        self.assertNotEqual(0, self._mean_squared_error(samples, new_samples))

    def test_can_phase_shift_1d_signal_180_degrees(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Seconds(1), [110, 220, 440, 880])
        coeffs = fft(samples)
        shifted = phase_shift(
            coeffs=coeffs,
            samplerate=samplerate,
            time_shift=-Milliseconds(1000),
            frequency_band=FrequencyBand(50, 5000))
        new_samples = np.fft.irfft(shifted, norm='ortho')
        self.assertAlmostEqual(
            0, self._mean_squared_error(samples, new_samples), 1)

    def test_2d_phase_shift_returns_correct_shape(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Milliseconds(2500), [220, 440, 880])
        windowsize = TimeSlice(duration=Milliseconds(200))
        stepsize = TimeSlice(duration=Milliseconds(100))
        _, windowed = samples.sliding_window_with_leftovers(
            windowsize=windowsize, stepsize=stepsize, dopad=True)
        coeffs = fft(windowed)
        shifted = phase_shift(
            coeffs=coeffs,
            samplerate=samplerate,
            time_shift=Milliseconds(40),
            frequency_band=FrequencyBand(50, 5000))
        self.assertEqual(coeffs.shape, shifted.shape)

    def test_can_phase_shift_2d_signal(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Milliseconds(2500), [220, 440, 880])
        windowsize = TimeSlice(duration=Milliseconds(200))
        stepsize = TimeSlice(duration=Milliseconds(100))
        _, windowed = samples.sliding_window_with_leftovers(
            windowsize=windowsize, stepsize=stepsize, dopad=True)
        coeffs = fft(windowed)
        shifted = phase_shift(coeffs, samplerate, Milliseconds(40))
        synth = FFTSynthesizer()
        new_samples = synth.synthesize(shifted).squeeze()
        self.assertNotEqual(0, self._mean_squared_error(samples, new_samples))

    def test_can_phase_shift_2d_signal_180_degrees(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Milliseconds(2500), [220, 440, 880])
        windowsize = TimeSlice(duration=Milliseconds(200))
        stepsize = TimeSlice(duration=Milliseconds(100))
        _, windowed = samples.sliding_window_with_leftovers(
            windowsize=windowsize, stepsize=stepsize, dopad=True)
        coeffs = fft(windowed)
        shifted = phase_shift(
            coeffs=coeffs,
            samplerate=samplerate,
            time_shift=Milliseconds(100))
        synth = FFTSynthesizer()
        new_samples = synth.synthesize(shifted).squeeze()
        self.assertAlmostEqual(
            0, self._mean_squared_error(samples, new_samples), 1)

    def test_raises_value_error_when_specified_axis_not_frequency_dim(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate) \
            .synthesize(Milliseconds(2500), [220, 440, 880])
        self.assertRaises(
            ValueError,
            lambda: phase_shift(samples, samplerate, Milliseconds(10)))


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