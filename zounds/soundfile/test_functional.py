import unittest2
from functional import resample
from zounds import AudioSamples
from zounds import SR11025
from zounds import SR44100
from zounds import Seconds
from zounds import SineSynthesizer, SilenceSynthesizer


class ResampleTests(unittest2.TestCase):
    def test_can_resample_audio(self):
        samplerate = SR44100()
        synth = SineSynthesizer(samplerate)
        samples = synth.synthesize(Seconds(1), [440, 880, 1760])
        encoded = samples.encode()
        new_samples = AudioSamples.from_file(encoded)
        resampled = resample(new_samples, SR11025())
        self.assertIsInstance(resampled, AudioSamples)
        self.assertEqual(int(SR11025()), len(resampled))

    def test_can_resample_stereo(self):
        samplerate = SR44100()
        synth = SineSynthesizer(samplerate)
        samples = synth.synthesize(Seconds(1), [440, 880, 1760])
        encoded = samples.stereo.encode()
        new_samples = AudioSamples.from_file(encoded)
        resampled = resample(new_samples, SR11025())
        self.assertIsInstance(resampled, AudioSamples)
        self.assertEqual(int(SR11025()), len(resampled))

    def test_resample_does_not_introduce_pops(self):
        samplerate = SR44100()
        synth = SilenceSynthesizer(samplerate)
        samples = synth.synthesize(Seconds(1))
        resampled = resample(samples, SR11025())
        self.assertEqual(0, resampled.max())