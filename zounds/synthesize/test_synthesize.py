import unittest2
import numpy as np
from synthesize import \
    ShortTimeTransformSynthesizer, SineSynthesizer, DCTSynthesizer
from zounds.timeseries import \
    SR22050, SR44100, HalfLapped, Seconds, TimeDimension, AudioSamples
from zounds.core import ArrayWithUnits
from zounds.spectral import FrequencyDimension, FrequencyBand, LinearScale


class SynthesizeTests(unittest2.TestCase):
    def test_has_correct_sample_rate(self):
        half_lapped = HalfLapped()
        # synth = ShortTimeTransformSynthesizer()
        synth = DCTSynthesizer()
        raw = np.zeros((100, 2048))
        band = FrequencyBand(0, SR44100().nyquist)
        scale = LinearScale(band, raw.shape[1])
        timeseries = ArrayWithUnits(
                raw, [TimeDimension(*half_lapped), FrequencyDimension(scale)])
        output = synth.synthesize(timeseries)
        self.assertIsInstance(output.samplerate, SR44100)
        self.assertIsInstance(output, AudioSamples)


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
