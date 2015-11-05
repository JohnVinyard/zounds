from __future__ import division
from duration import Picoseconds, Seconds
import unittest2

class SampleRate(object):
    
    def __init__(self, frequency, duration):
        self.frequency = frequency
        self.duration = duration
        super(SampleRate, self).__init__()
    
    @property
    def overlap(self):
        return self.duration - self.frequency
    
    def __mul__(self, other):
        try:
            if len(other) == 1:
                other = other * 2
        except TypeError:
            other = (other, other)
        
        freq = self.frequency * other[0]
        duration = (self.frequency * other[1]) + self.overlap
        return SampleRate(freq, duration)

class AudioSampleRate(SampleRate):
    
    def __init__(self, samples_per_second):
        one_sample = Picoseconds(int(1e12)) // samples_per_second
        super(AudioSampleRate, self).__init__(one_sample, one_sample)
    
    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)
    
class SR96000(AudioSampleRate):
    
    def __init__(self):
        super(SR96000, self).__init__(96000)
        
class SR48000(AudioSampleRate):
    
    def __init__(self):
        super(SR48000, self).__init__(48000)
        
class SR44100(AudioSampleRate):
    
    def __init__(self):
        super(SR44100, self).__init__(44100)

class SR22050(AudioSampleRate):
    
    def __init__(self):
        super(SR22050, self).__init__(22050)

class SR11025(AudioSampleRate):
    
    def __init__(self):
        super(SR11025, self).__init__(11025)

class HalfLapped(SampleRate):
    
    def __init__(self, window_at_44100 = 2048, hop_at_44100 = 1024):
        one_sample_at_44100 = Picoseconds(int(1e12)) / 44100.
        window = one_sample_at_44100 * window_at_44100
        step = one_sample_at_44100 * hop_at_44100
        super(HalfLapped, self).__init__(step, window) 

class SampleRateTests(unittest2.TestCase):
    
    def test_sr_96000_frequency(self):
        self.assertEqual(96000, SR96000().samples_per_second)
    
    def test_sr_48000_frequency(self):
        self.assertEqual(48000, SR48000().samples_per_second)
    
    def test_sr_44100_frequency(self):
        self.assertEqual(44100, SR44100().samples_per_second)
    
    def test_sr_22050_frequency(self):
        self.assertEqual(22050, SR22050().samples_per_second)
    
    def test_sr_11025_frequency(self):
        self.assertEqual(11025, SR11025().samples_per_second)
    
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
        