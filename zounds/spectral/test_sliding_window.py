from sliding_window import SlidingWindow
from zounds.timeseries import \
    AudioSamples, SR22050, SR44100, SR11025, SR48000, SR96000
import numpy as np
import unittest2


class SlidingWindowTests(unittest2.TestCase):

    def _check(self, samplerate, expected_window_size, expected_step_size):
        sw = SlidingWindow(wscheme=samplerate.half_lapped())
        samples = AudioSamples(
                np.zeros(5 * samplerate.samples_per_second), samplerate)
        sw._enqueue(samples, None)
        self.assertEqual(expected_window_size, sw._windowsize)
        self.assertEqual(expected_step_size, sw._stepsize)

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
