from .resample import Resample
from zounds.timeseries import SR44100, SR11025, Seconds
from zounds.synthesize import SilenceSynthesizer
from multiprocessing.pool import ThreadPool
import unittest2


class ResampleTests(unittest2.TestCase):

    def test_can_do_multithreaded_resampling(self):
        synth = SilenceSynthesizer(SR44100())
        audio = [synth.synthesize(Seconds(5)) for _ in range(10)]
        pool = ThreadPool(4)

        def x(samples):
            rs = Resample(int(SR44100()), int(SR11025()))
            return rs(samples, end_of_input=True)

        resampled = pool.map(x, audio)
        self.assertEqual(10, len(resampled))

    def test_correct_output_with_stereo(self):
        synth = SilenceSynthesizer(SR44100())
        samples = synth.synthesize(Seconds(1)).stereo
        rs = Resample(int(samples.samplerate), int(SR11025()), nchannels=2)
        resampled = rs(samples, end_of_input=True)
        self.assertEqual((11025, 2), resampled.shape)