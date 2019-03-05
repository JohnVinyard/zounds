import numpy as np
import unittest2
from .functional import \
    fft, stft, apply_scale, frequency_decomposition, phase_shift, rainbowgram, \
    fir_filter_bank, auto_correlogram, time_stretch, pitch_shift, \
    morlet_filter_bank
from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.synthesize import \
    SilenceSynthesizer, TickSynthesizer, SineSynthesizer, FFTSynthesizer
from zounds.timeseries import SR22050, Seconds, Milliseconds, TimeDimension, \
    TimeSlice, AudioSamples
from zounds.spectral import \
    HanningWindowingFunc, FrequencyDimension, LinearScale, GeometricScale, \
    ExplicitFrequencyDimension, FrequencyBand, MelScale
from matplotlib import cm


class FIRFilterBankTests(unittest2.TestCase):
    def test_has_correct_dimensions(self):
        samplerate = SR22050()
        scale = GeometricScale(
            start_center_hz=20,
            stop_center_hz=10000,
            bandwidth_ratio=0.2,
            n_bands=100)
        scale.ensure_overlap_ratio(0.5)
        taps = 256
        filter_bank = fir_filter_bank(scale, taps, samplerate, np.hanning(100))
        self.assertEqual((len(scale), taps), filter_bank.shape)
        self.assertEqual(FrequencyDimension(scale), filter_bank.dimensions[0])
        self.assertEqual(TimeDimension(*samplerate), filter_bank.dimensions[1])


class MorletFilterBankTests(unittest2.TestCase):

    def test_raises_when_scale_factors_length_does_not_match_scale(self):
        sr = SR22050()
        band = FrequencyBand(1, sr.nyquist)
        scale = MelScale(band, 512)
        scale_factors = np.linspace(0.1, 1.0, len(scale) // 2)
        self.assertRaises(
            ValueError,
            lambda: morlet_filter_bank(sr, 512, scale, scale_factors))

    def test_raises_when_scale_factors_is_not_a_collection_or_float(self):
        sr = SR22050()
        band = FrequencyBand(1, sr.nyquist)
        scale = MelScale(band, 512)
        scale_factors = object()
        self.assertRaises(
            TypeError,
            lambda: morlet_filter_bank(sr, 512, scale, scale_factors))

    def test_dimensions_are_correct(self):
        sr = SR22050()
        band = FrequencyBand(1, sr.nyquist)
        scale = MelScale(band, 128)
        scale_factors = np.linspace(0.1, 1.0, len(scale))
        filter_bank = morlet_filter_bank(sr, 512, scale, scale_factors)
        self.assertEqual((128, 512), filter_bank.shape)
        expected_freq_dimension = FrequencyDimension(scale)
        expected_time_dimension = TimeDimension(*sr)
        self.assertEqual(expected_freq_dimension, filter_bank.dimensions[0])
        self.assertEqual(expected_time_dimension, filter_bank.dimensions[1])

    def test_filters_are_normalized(self):
        sr = SR22050()
        band = FrequencyBand(1, sr.nyquist)
        scale = MelScale(band, 128)
        scale_factors = np.linspace(0.1, 1.0, len(scale))
        filter_bank = morlet_filter_bank(
            sr, 512, scale, scale_factors, normalize=True)
        norms = np.linalg.norm(filter_bank, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

class AutoCorrelogramTests(unittest2.TestCase):
    @unittest2.skip
    def test_smoke(self):
        samples = AudioSamples.silence(SR22050(), Seconds(1))
        samplerate = SR22050()
        scale = GeometricScale(
            start_center_hz=20,
            stop_center_hz=5000,
            bandwidth_ratio=1.2,
            n_bands=8)
        scale.ensure_overlap_ratio(0.5)
        taps = 16
        filter_bank = fir_filter_bank(scale, taps, samplerate, np.hanning(3))
        correlogram = auto_correlogram(samples, filter_bank)
        self.assertEqual(3, correlogram.ndim)


class FrequencyDecompositionTests(unittest2.TestCase):
    def test_can_decompose_audio_samples(self):
        samples = AudioSamples.silence(SR22050(), Seconds(1))
        bands = frequency_decomposition(samples, [64, 128, 256, 512, 1024])
        expected_td = TimeDimension(samples.end, samples.end)
        self.assertEqual(expected_td, bands.dimensions[0])
        self.assertIsInstance(bands.dimensions[1], ExplicitFrequencyDimension)

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

    def test_can_take_stft_of_batch(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(6666))
        stacked = ArrayWithUnits(
            np.zeros((10,) + samples.shape, dtype=samples.dtype),
            (IdentityDimension(),) + samples.dimensions)
        stacked[:] = samples
        wscheme = sr.windowing_scheme(512, 256)
        tf = stft(stacked, wscheme, HanningWindowingFunc())

        self.assertEqual(10, len(tf))
        self.assertIsInstance(tf, ArrayWithUnits)
        self.assertEqual(3, len(tf.dimensions))
        self.assertIsInstance(tf.dimensions[0], IdentityDimension)
        self.assertIsInstance(tf.dimensions[1], TimeDimension)
        self.assertEqual(tf.dimensions[1].samplerate, wscheme)
        self.assertIsInstance(tf.dimensions[2], FrequencyDimension)
        self.assertIsInstance(tf.dimensions[2].scale, LinearScale)

    def test_stft_raises_for_invalid_dimensions(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(6666))
        wscheme = sr.windowing_scheme(512, 256)
        tf = stft(samples, wscheme, HanningWindowingFunc())
        self.assertRaises(
            ValueError, lambda: stft(tf, wscheme, HanningWindowingFunc()))


class PhaseStretchTests(unittest2.TestCase):
    def test_can_pitch_shift_audio_samples(self):
        sr = SR22050()
        samples = SineSynthesizer(sr).synthesize(Milliseconds(6666), [440])
        shifted = pitch_shift(samples, 1.0).squeeze()
        self.assertEqual(len(samples), len(shifted))

    def test_can_pitch_shift_batch(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(6666))
        stacked = ArrayWithUnits(
            np.zeros((10,) + samples.shape, dtype=samples.dtype),
            (IdentityDimension(),) + samples.dimensions)
        stacked[:] = samples
        stretched = pitch_shift(stacked, -2.0)
        self.assertEqual(10, stretched.shape[0])
        self.assertEqual(len(samples), stretched.shape[1])


class TimeStretchTests(unittest2.TestCase):
    def test_can_stretch_audio_samples(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(1000))
        stretched = time_stretch(samples, 0.5).squeeze()
        self.assertEqual(int(2 * len(samples)), len(stretched))

    def test_can_contract_audio_samples(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(1000))
        print('First',samples.shape, samples.dimensions)
        stretched = time_stretch(samples, 2.0).squeeze()
        print('Second',stretched.shape, stretched.dimensions)
        self.assertEqual(len(samples) // 2, len(stretched))

    def test_can_stretch_audio_batch(self):
        sr = SR22050()
        samples = SilenceSynthesizer(sr).synthesize(Milliseconds(6666))
        stacked = ArrayWithUnits(
            np.zeros((10,) + samples.shape, dtype=samples.dtype),
            (IdentityDimension(),) + samples.dimensions)
        stacked[:] = samples
        stretched = time_stretch(stacked, 2.0)
        self.assertEqual(10, stretched.shape[0])
        self.assertEqual(int(len(samples) // 2), stretched.shape[1])


class RainbowgramTests(unittest2.TestCase):
    def test_should_have_correct_shape_and_dimensions(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate).synthesize(Milliseconds(8888))
        wscheme = samplerate.windowing_scheme(256, 128)
        tf = stft(samples, wscheme, HanningWindowingFunc())
        rg = rainbowgram(tf, cm.rainbow)
        self.assertEqual(3, rg.ndim)
        self.assertEqual(tf.shape + (3,), rg.shape)
        self.assertEqual(3, len(rg.dimensions))
        self.assertEqual(tf.dimensions[0], rg.dimensions[0])
        self.assertEqual(tf.dimensions[1], rg.dimensions[1])
        self.assertEqual(rg.dimensions[2], IdentityDimension())
        self.assertEqual(3, rg.shape[-1])


class ApplyScaleTests(unittest2.TestCase):
    def test_apply_scale_to_self_is_identity_function(self):
        samplerate = SR22050()
        samples = SineSynthesizer(samplerate).synthesize(Milliseconds(8888))
        wscheme = samplerate.windowing_scheme(256, 128)
        tf = stft(samples, wscheme, HanningWindowingFunc())
        scale = tf.dimensions[-1].scale
        transformed = apply_scale(tf, scale, HanningWindowingFunc())
        self.assertEqual(tf.shape, transformed.shape)
        self.assertEqual(tf.dimensions[0], transformed.dimensions[0])
        self.assertEqual(tf.dimensions[1], transformed.dimensions[1])

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
