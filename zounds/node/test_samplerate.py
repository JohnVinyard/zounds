import unittest2
from duration import Seconds
from samplerate import \
    SampleRate, SR96000, SR48000, SR44100, SR22050, SR11025, audio_sample_rate


class SampleRateTests(unittest2.TestCase):
    def test_raises_for_unknown_audio_samplerate(self):
        self.assertRaises(ValueError, lambda: audio_sample_rate(1))

    def test_sr_96000_frequency(self):
        self.assertEqual(96000, SR96000().samples_per_second)

    def test_get_96000_frequency(self):
        self.assertIsInstance(audio_sample_rate(96000), SR96000)

    def test_sr_48000_frequency(self):
        self.assertEqual(48000, SR48000().samples_per_second)

    def test_get_48000_frequency(self):
        self.assertIsInstance(audio_sample_rate(48000), SR48000)

    def test_sr_44100_frequency(self):
        self.assertEqual(44100, SR44100().samples_per_second)

    def test_get_44100_frequency(self):
        self.assertIsInstance(audio_sample_rate(44100), SR44100)

    def test_sr_22050_frequency(self):
        self.assertEqual(22050, SR22050().samples_per_second)

    def test_get_22050_freuency(self):
        self.assertIsInstance(audio_sample_rate(22050), SR22050)

    def test_sr_11025_frequency(self):
        self.assertEqual(11025, SR11025().samples_per_second)

    def test_get_11025_frequency(self):
        self.assertIsInstance(audio_sample_rate(11025), SR11025)

    def test_no_overlap(self):
        self.assertEqual(Seconds(0), SampleRate(Seconds(1), Seconds(1)).overlap)

    def test_some_overlap(self):
        self.assertEqual(Seconds(1), SampleRate(Seconds(1), Seconds(2)).overlap)

    def test_multiply_no_overlap_number(self):
        sr = SampleRate(Seconds(1), Seconds(1)) * 2
        self.assertEqual(Seconds(2), sr.frequency)
        self.assertEqual(Seconds(2), sr.duration)

    def test_multiply_some_overlap_number(self):
        sr = SampleRate(Seconds(1), Seconds(2)) * 2
        self.assertEqual(Seconds(2), sr.frequency)
        self.assertEqual(Seconds(3), sr.duration)

    def test_multiply_no_overlap_single_value(self):
        sr = SampleRate(Seconds(1), Seconds(1)) * (2,)
        self.assertEqual(Seconds(2), sr.frequency)
        self.assertEqual(Seconds(2), sr.duration)

    def test_multiply_some_overlap_single_value(self):
        sr = SampleRate(Seconds(1), Seconds(2)) * (2,)
        self.assertEqual(Seconds(2), sr.frequency)
        self.assertEqual(Seconds(3), sr.duration)

    def test_multiply_no_overlap_two_values(self):
        sr = SampleRate(Seconds(1), Seconds(1)) * (2, 4)
        self.assertEqual(Seconds(2), sr.frequency)
        self.assertEqual(Seconds(4), sr.duration)

    def test_multiply_some_overlap_two_values(self):
        sr = SampleRate(Seconds(1), Seconds(2)) * (2, 4)
        self.assertEqual(Seconds(2), sr.frequency)
        self.assertEqual(Seconds(5), sr.duration)
