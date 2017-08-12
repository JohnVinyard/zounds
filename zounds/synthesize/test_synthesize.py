import numpy as np
import unittest2

from synthesize import \
    SineSynthesizer, DCTSynthesizer, FFTSynthesizer, NoiseSynthesizer, \
    SilenceSynthesizer
from zounds.basic import stft, resampled
from zounds.core import ArrayWithUnits
from zounds.persistence import ArrayWithUnitsFeature
from zounds.spectral import \
    FrequencyDimension, FrequencyBand, LinearScale, FFT, SlidingWindow, \
    OggVorbisWindowingFunc
from zounds.timeseries import \
    SR22050, SR44100, SR11025, SR48000, SR96000, HalfLapped, Seconds, \
    TimeDimension, AudioSamples, SampleRate, Milliseconds
from zounds.util import simple_in_memory_settings


class SynthesizeTests(unittest2.TestCase):
    def test_has_correct_sample_rate(self):
        half_lapped = HalfLapped()
        synth = DCTSynthesizer()
        raw = np.zeros((100, 2048))
        band = FrequencyBand(0, SR44100().nyquist)
        scale = LinearScale(band, raw.shape[1])
        timeseries = ArrayWithUnits(
            raw, [TimeDimension(*half_lapped), FrequencyDimension(scale)])
        output = synth.synthesize(timeseries)
        self.assertIsInstance(output.samplerate, SR44100)
        self.assertIsInstance(output, AudioSamples)


class FFTSynthesizerTests(unittest2.TestCase):
    def can_invert_fft(self, samplerate):
        base_cls = stft(
            resample_to=samplerate,
            store_fft=True,
            store_windowed=True)

        @simple_in_memory_settings
        class Document(base_cls):
            pass

        synth = SineSynthesizer(samplerate)
        audio = synth.synthesize(Seconds(2), freqs_in_hz=[440., 880.])

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        fft_synth = FFTSynthesizer()
        recon = fft_synth.synthesize(doc.fft)

        self.assertIsInstance(recon, ArrayWithUnits)
        self.assertEqual(audio.dimensions, recon.dimensions)

    def test_can_invert_long_fft(self):
        samplerate = SR11025()
        rs = resampled(resample_to=samplerate)

        @simple_in_memory_settings
        class Document(rs):
            long_windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=SampleRate(
                    Milliseconds(500),
                    Seconds(1)),
                wfunc=OggVorbisWindowingFunc(),
                needs=rs.resampled,
                store=True)

            long_fft = ArrayWithUnitsFeature(
                FFT,
                needs=long_windowed,
                store=True)

        synth = SineSynthesizer(samplerate)
        audio = synth.synthesize(Seconds(2), freqs_in_hz=[440., 880.])

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        fft_synth = FFTSynthesizer()
        recon = fft_synth.synthesize(doc.long_fft)
        self.assertIsInstance(recon, AudioSamples)
        self.assertEqual(audio.dimensions, recon.dimensions)

    def test_can_invert_fft_11025(self):
        self.can_invert_fft(SR11025())

    def test_can_invert_fft_22050(self):
        self.can_invert_fft(SR22050())

    def test_can_invert_fft_44100(self):
        self.can_invert_fft(SR44100())

    def test_can_invert_fft_48000(self):
        self.can_invert_fft(SR48000())

    @unittest2.skip(
        'HalfLapped does not compute the right window size in samples')
    def test_can_invert_fft_96000(self):
        self.can_invert_fft(SR96000())


class SineSynthesizerTests(unittest2.TestCase):
    def test_generates_correct_shape(self):
        ss = SineSynthesizer(SR22050())
        audio = ss.synthesize(Seconds(4), freqs_in_hz=[440.])
        self.assertEqual(1, len(audio.shape))

    def test_generates_correct_samplerate(self):
        ss = SineSynthesizer(SR44100())
        audio = ss.synthesize(Seconds(4), freqs_in_hz=[440.])
        self.assertEqual(SR44100(), audio.samplerate)

    def test_generates_correct_number_of_samples(self):
        samplerate = SR22050()
        duration = Seconds(1)
        ss = SineSynthesizer(samplerate)
        audio = ss.synthesize(duration, freqs_in_hz=[440.])
        expected_samples = int(samplerate) * int(duration / Seconds(1))
        self.assertEqual(expected_samples, len(audio))

    def test_can_create_audio_with_single_tone(self):
        ss = SineSynthesizer(SR22050())
        audio = ss.synthesize(Seconds(4), freqs_in_hz=[440.])
        fft = abs(np.fft.rfft(audio))
        self.assertEqual(1, (fft > 1).sum())

    def test_can_create_audio_with_multiple_tones(self):
        ss = SineSynthesizer(SR22050())
        audio = ss.synthesize(Seconds(4), freqs_in_hz=[440., 660.])
        fft = abs(np.fft.rfft(audio))
        self.assertEqual(2, (fft > 1).sum())


class NoiseSynthesizerTests(unittest2.TestCase):
    def test_noise_synth_outputs_values_in_correct_range(self):
        ns = NoiseSynthesizer(SR11025())
        audio = ns.synthesize(Seconds(1))
        self.assertLess(audio.min(), 0)
        self.assertGreater(audio.max(), 0)


class SilenceSynthesizerTests(unittest2.TestCase):
    def test_silence_synthesizer_outputs_zero(self):
        synth = SilenceSynthesizer(SR11025())
        audio = synth.synthesize(Seconds(1))
        np.testing.assert_allclose(audio, 0)