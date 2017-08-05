import numpy as np
import scipy
import unittest2

from frequencyscale import GeometricScale
from zounds.basic import resampled, stft
from zounds.core import ArrayWithUnits
from zounds.persistence import ArrayWithUnitsFeature, FrequencyAdaptiveFeature
from zounds.spectral import \
    SlidingWindow, DCTIV, MDCT, FFT, SpectralCentroid, OggVorbisWindowingFunc, \
    SpectralFlatness, FrequencyAdaptiveTransform, DCT
from zounds.synthesize import \
    SineSynthesizer, DCTIVSynthesizer, MDCTSynthesizer, NoiseSynthesizer, \
    TickSynthesizer
from zounds.timeseries import \
    SR11025, SR22050, SR44100, Seconds, Milliseconds, Picoseconds, \
    AudioSamples, TimeSlice, TimeDimension
from zounds.timeseries.samplerate import SampleRate, HalfLapped
from zounds.util import simple_in_memory_settings


class FrequencyAdaptiveTransformTests(unittest2.TestCase):

    def test_square_form_with_overlap_add(self):
        samplerate = SR11025()
        BaseModel = stft(resample_to=samplerate)
        windowing_func = OggVorbisWindowingFunc()
        scale = GeometricScale(20, 5000, 0.1, 25)

        @simple_in_memory_settings
        class Document(BaseModel):
            long_windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=SampleRate(
                    frequency=Milliseconds(500),
                    duration=Seconds(1)),
                wfunc=windowing_func,
                needs=BaseModel.resampled,
                store=True)

            dct = ArrayWithUnitsFeature(
                DCT,
                scale_always_even=True,
                needs=long_windowed,
                store=True)

            mdct = FrequencyAdaptiveFeature(
                FrequencyAdaptiveTransform,
                transform=scipy.fftpack.idct,
                scale=scale,
                needs=dct,
                store=True)

        synth = TickSynthesizer(SR22050())
        samples = synth.synthesize(Seconds(5), Milliseconds(200))
        _id = Document.process(meta=samples.encode())
        doc = Document(_id)
        square = doc.mdct.square(30, do_overlap_add=True)
        self.assertEqual(2, square.ndim)
        self.assertEqual(150, square.shape[0])
        self.assertEqual(25, square.shape[1])

    def test_square_form_no_overlap_add(self):
        samplerate = SR11025()
        BaseModel = stft(resample_to=samplerate)
        windowing_func = OggVorbisWindowingFunc()
        scale = GeometricScale(20, 5000, 0.1, 25)

        @simple_in_memory_settings
        class Document(BaseModel):
            long_windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=SampleRate(
                    frequency=Milliseconds(500),
                    duration=Seconds(1)),
                wfunc=windowing_func,
                needs=BaseModel.resampled,
                store=True)

            dct = ArrayWithUnitsFeature(
                DCT,
                scale_always_even=True,
                needs=long_windowed,
                store=True)

            mdct = FrequencyAdaptiveFeature(
                FrequencyAdaptiveTransform,
                transform=scipy.fftpack.idct,
                scale=scale,
                needs=dct,
                store=True)

        synth = TickSynthesizer(SR22050())
        samples = synth.synthesize(Seconds(5), Milliseconds(200))
        _id = Document.process(meta=samples.encode())
        doc = Document(_id)
        square = doc.mdct.square(30)
        self.assertEqual(3, square.ndim)
        self.assertEqual(30, square.shape[1])
        self.assertEqual(25, square.shape[2])


class SpectralFlatnessTests(unittest2.TestCase):
    def setUp(self):
        self.samplerate = SR22050()
        rs = resampled(resample_to=self.samplerate)

        wscheme = HalfLapped()

        @simple_in_memory_settings
        class Document(rs):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=wscheme,
                wfunc=OggVorbisWindowingFunc(),
                needs=rs.resampled,
                store=False)

            fft = ArrayWithUnitsFeature(
                FFT,
                needs=windowed,
                store=False)

            flatness = ArrayWithUnitsFeature(
                SpectralFlatness,
                needs=fft,
                store=True)

        # create a pure sine wave that fades out
        ss = SineSynthesizer(self.samplerate)
        sine = ss.synthesize(Seconds(5), [440.])
        sine_envelope = np.linspace(1.0, 0.0, len(sine))
        sine *= sine_envelope

        # create noise
        ns = NoiseSynthesizer(self.samplerate)
        noise = ns.synthesize(Seconds(5))
        noise_envelope = np.linspace(0.0, 1.0, len(noise))
        noise *= noise_envelope

        # mix the sine wave and noise together
        self.audio = sine + noise

        _id = Document.process(meta=self.audio.encode())
        self.doc = Document(_id)

    def test_has_correct_type(self):
        self.assertIsInstance(self.doc.flatness, ArrayWithUnits)

    def test_has_correct_duration(self):
        self.assertAlmostEqual(
            self.audio.dimensions[0].end_seconds,
            self.doc.flatness.dimensions[0].end_seconds,
            delta=0.02)

    def test_has_correct_dimensions(self):
        self.assertEqual(1, len(self.doc.flatness.dimensions))
        self.assertIsInstance(self.doc.flatness.dimensions[0], TimeDimension)

    def test_flatness_is_monotonically_increasing(self):
        chunked = self.doc.flatness \
            .sliding_window((TimeSlice(Seconds(1)),)) \
            .mean(axis=1)
        diff = np.diff(chunked)
        self.assertTrue(np.all(diff >= 0))


class SpectralCentroidTests(unittest2.TestCase):
    def setUp(self):
        self.samplerate = SR44100()
        rs = resampled(resample_to=self.samplerate)

        wscheme = HalfLapped()

        @simple_in_memory_settings
        class Document(rs):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=wscheme,
                wfunc=OggVorbisWindowingFunc(),
                needs=rs.resampled,
                store=False)

            fft = ArrayWithUnitsFeature(
                FFT,
                needs=windowed,
                store=False)

            centroid = ArrayWithUnitsFeature(
                SpectralCentroid,
                needs=fft,
                store=True)

        ss = SineSynthesizer(self.samplerate)
        chunks = \
            [ss.synthesize(Seconds(1), [440 * i]) for i in xrange(1, 6)]
        self.audio = \
            AudioSamples(ArrayWithUnits.concat(chunks), self.samplerate)

        _id = Document.process(meta=self.audio.encode())
        self.doc = Document(_id)

    def test_has_correct_type(self):
        self.assertIsInstance(self.doc.centroid, ArrayWithUnits)

    def test_has_correct_dimensions(self):
        self.assertEqual(1, len(self.doc.centroid.dimensions))

    def test_has_correct_duration(self):
        self.assertAlmostEqual(
            self.audio.dimensions[0].end_seconds,
            self.doc.centroid.dimensions[0].end_seconds,
            delta=0.02)

    def test_centroid_is_monotonically_increasing(self):
        chunked = self.doc.centroid \
            .sliding_window((TimeSlice(Seconds(1)),)) \
            .mean(axis=1)
        diff = np.diff(chunked)
        self.assertTrue(np.all(diff >= 0))


class MDCTTests(unittest2.TestCase):
    def setUp(self):
        self.samplerate = SR11025()
        rs = resampled(resample_to=self.samplerate)

        wscheme = HalfLapped()

        @simple_in_memory_settings
        class Document(rs):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=wscheme,
                needs=rs.resampled,
                store=False)

            mdct = ArrayWithUnitsFeature(
                MDCT,
                needs=windowed,
                store=True)

        ss = SineSynthesizer(self.samplerate)
        self.audio = ss.synthesize(Seconds(5), [440., 660., 880.])

        _id = Document.process(meta=self.audio.encode())
        self.doc = Document(_id)

    def test_has_correct_duration(self):
        self.assertAlmostEqual(
            self.audio.dimensions[0].end_seconds,
            self.doc.mdct.dimensions[0].end_seconds,
            delta=0.02)

    def test_perfect_reconstruction_using_overlap_add(self):
        synth = SineSynthesizer(SR22050())
        audio = synth.synthesize(Seconds(10), [440., 660., 880.])
        windowed = audio.sliding_window(
            (TimeSlice(Seconds(1)),),
            (TimeSlice(Milliseconds(500)),))

        mdct = MDCT()

        coeffs = list(mdct._process(windowed * OggVorbisWindowingFunc()))[0]

        mdct_synth = MDCTSynthesizer()
        recon = mdct_synth.synthesize(coeffs)

        # take a slice, so we can ignore boundary conditions
        slce = TimeSlice(start=Seconds(1), duration=Seconds(8))

        np.testing.assert_almost_equal(recon[slce], audio[slce], decimal=4)

    def test_is_correct_type(self):
        self.assertIsInstance(self.doc.mdct, ArrayWithUnits)

    def test_has_correct_nyquist_frequency(self):
        freq_dim = self.doc.mdct.dimensions[1]
        self.assertEqual(self.samplerate.nyquist, freq_dim.scale.stop_hz)

    def test_reconstruction(self):
        ds = MDCTSynthesizer()
        recon = ds.synthesize(self.doc.mdct)

        orig_seconds = self.audio.dimensions[0].end_seconds
        recon_seconds = recon.dimensions[0].end_seconds
        self.assertAlmostEqual(orig_seconds, recon_seconds, delta=0.02)

        # ensure that both the original and reconstruction have the exact
        # same length in samples, so that we can easily compare spectral peaks
        l = min(len(self.audio), len(recon))

        self.assertEqual(self.audio.samplerate, recon.samplerate)
        orig_fft = abs(np.fft.rfft(self.audio[:l]))
        recon_fft = abs(np.fft.rfft(recon[:l]))
        orig_peaks = set(np.argsort(orig_fft)[-3:])
        recon_peaks = set(np.argsort(recon_fft)[-3:])
        # ensure that the original and reconstruction have the same three
        # spectral peaks
        self.assertEqual(orig_peaks, recon_peaks)


class DCTIVTests(unittest2.TestCase):
    def setUp(self):
        self.samplerate = SR22050()
        rs = resampled(resample_to=self.samplerate)

        window_size = Picoseconds(int(1e12))
        wscheme = SampleRate(window_size, window_size)

        @simple_in_memory_settings
        class Document(rs):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=wscheme,
                needs=rs.resampled,
                store=False)

            dct = ArrayWithUnitsFeature(
                DCTIV,
                needs=windowed,
                store=True)

        ss = SineSynthesizer(self.samplerate)
        self.audio = ss.synthesize(Seconds(5), [440., 660., 880.])

        _id = Document.process(meta=self.audio.encode())
        self.doc = Document(_id)

    @unittest2.skip
    def test_perfect_reconstruction(self):
        synth = SineSynthesizer(SR22050())
        audio = synth.synthesize(Seconds(1), [440., 660., 880.])
        node = DCTIV()
        coeffs = node._process_raw(audio[None, :])
        recon = node._process_raw(coeffs)[0]

        # see? it's do-able w/ dct
        dct_coeffs = scipy.fftpack.dct(audio[None, :], norm='ortho')
        inverse_dct = scipy.fftpack.idct(dct_coeffs, norm='ortho')[0]

        np.testing.assert_almost_equal(inverse_dct, audio, decimal=4)
        np.testing.assert_almost_equal(recon, audio, decimal=4)

    def test_is_correct_type(self):
        self.assertIsInstance(self.doc.dct, ArrayWithUnits)

    def test_has_correct_nyquist_frequency(self):
        freq_dim = self.doc.dct.dimensions[-1]
        self.assertEqual(self.samplerate.nyquist, freq_dim.scale.stop_hz)

    def test_reconstruction(self):
        ds = DCTIVSynthesizer()
        recon = ds.synthesize(self.doc.dct)

        orig_seconds = self.audio.dimensions[0].end_seconds
        recon_seconds = recon.dimensions[0].end_seconds
        self.assertAlmostEqual(orig_seconds, recon_seconds, delta=0.02)

        # ensure that both the original and reconstruction have the exact
        # same length in samples, so that we can easily compare spectral peaks
        l = min(len(self.audio), len(recon))

        self.assertEqual(self.audio.samplerate, recon.samplerate)
        orig_fft = abs(np.fft.rfft(self.audio[:l]))
        recon_fft = abs(np.fft.rfft(recon[:l]))
        orig_peaks = set(np.argsort(orig_fft)[-3:])
        recon_peaks = set(np.argsort(recon_fft)[-3:])
        # ensure that the original and reconstruction have the same three
        # spectral peaks
        self.assertEqual(orig_peaks, recon_peaks)
