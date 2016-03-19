import unittest2
import numpy as np
from synthesize import ShortTimeTransformSynthesizer
from zounds.timeseries import \
    ConstantRateTimeSeries, SR44100, HalfLapped, AudioSamples


class SynthesizeTests(unittest2.TestCase):
    def test_has_correct_sample_rate(self):
        half_lapped = HalfLapped()
        synth = ShortTimeTransformSynthesizer()
        raw = np.zeros((100, 2048))
        timeseries = ConstantRateTimeSeries(
                raw, half_lapped.frequency, half_lapped.duration)
        output = synth.synthesize(timeseries)
        self.assertIsInstance(output.samplerate, SR44100)
        self.assertIsInstance(output, AudioSamples)
