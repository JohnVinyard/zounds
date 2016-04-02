from __future__ import division
from duration import Picoseconds
from collections import namedtuple

Stride = namedtuple('Stride', ['frequency', 'duration'])


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
                other *= 2
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


_samplerates = (SR96000(), SR48000(), SR44100(), SR22050(), SR11025())


def audio_sample_rate(samples_per_second):
    for sr in _samplerates:
        if samples_per_second == sr.samples_per_second:
            return sr
    raise ValueError(
            '{samples_per_second} is an invalid sample rate'.format(**locals()))


class HalfLapped(SampleRate):
    def __init__(self, window_at_44100=2048, hop_at_44100=1024):
        one_sample_at_44100 = Picoseconds(int(1e12)) / 44100.
        window = one_sample_at_44100 * window_at_44100
        step = one_sample_at_44100 * hop_at_44100
        super(HalfLapped, self).__init__(step, window)
