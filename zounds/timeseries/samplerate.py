from __future__ import division
from duration import Picoseconds
from collections import namedtuple
import numpy as np
from duration import Seconds

Stride = namedtuple('Stride', ['frequency', 'duration'])


class SampleRate(object):
    """
    `SampleRate` describes the constant frequency at which samples are taken
    from a continuous signal, and the duration of each sample.

    Instances of this class could describe an audio sampling rate (e.g. 44.1kHz)
    or the strided windows often used in short-time fourier transforms

    Args:
        frequency (numpy.timedelta64): The frequency at which the signal is
            sampled
        duration (numpy.timedelta64): The duration of each sample

    Raises:
        ValueError: when frequency or duration are less than or equal to zero

    Examples:
        >>> from zounds import Seconds, SampleRate
        >>> sr = SampleRate(Seconds(1), Seconds(2))
        >>> sr.frequency
        numpy.timedelta64(1,'s')
        >>> sr.duration
        numpy.timedelta64(2,'s')
        >>> sr.overlap
        numpy.timedelta64(1,'s')
        >>> sr.overlap_ratio
        0.5

    See Also:
        :class:`SR96000`
        :class:`SR48000`
        :class:`SR44100`
        :class:`SR22050`
        :class:`SR11025`
    """

    def __init__(self, frequency, duration):
        if frequency.astype(np.int) <= 0:
            raise ValueError('frequency must be positive')
        if duration.astype(np.int) <= 0:
            raise ValueError('duration must be positive')

        self.frequency = frequency
        self.duration = duration
        super(SampleRate, self).__init__()

    def __str__(self):
        f = self.frequency / Seconds(1)
        d = self.duration / Seconds(1)
        return '{self.__class__.__name__}(f={f}, d={d})'.format(**locals())

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter((self.frequency, self.duration))

    def __len__(self):
        return 2

    def __eq__(self, other):
        return \
            self.frequency == other.frequency \
            and self.duration == other.duration

    @property
    def overlap(self):
        """
        For sampling schemes that overlap, return a :class:`numpy.timedelta64`
        instance representing the duration of overlap between each sample
        """
        return self.duration - self.frequency

    @property
    def overlap_ratio(self):
        """
        For sampling schemes that overlap, return the ratio of overlap to
        sample duration
        """
        return self.overlap / self.duration

    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)

    @property
    def nyquist(self):
        return self.samples_per_second // 2

    def __mul__(self, other):
        try:
            if len(other) == 1:
                other *= 2
        except TypeError:
            other = (other, other)

        freq = self.frequency * other[0]
        duration = (self.frequency * other[1]) + self.overlap
        new = SampleRate(freq, duration)
        return new

    def discrete_samples(self, ts):
        td = next(dim for dim in ts.dimensions if hasattr(dim, 'frequency'))
        windowsize = np.round((self.duration - td.overlap) / td.frequency)
        stepsize = np.round(self.frequency / td.frequency)
        return int(stepsize), int(windowsize)

    def resample(self, ratio):
        orig_freq = Picoseconds(int(self.frequency / Picoseconds(1)))
        orig_duration = Picoseconds(int(self.duration / Picoseconds(1)))
        f = orig_freq * ratio
        d = orig_duration * ratio
        return SampleRate(f, d)


class AudioSampleRate(SampleRate):
    def __init__(self, samples_per_second, suggested_window, suggested_hop):
        self.suggested_hop = suggested_hop
        self.suggested_window = suggested_window
        self.one_sample = Picoseconds(int(1e12)) // samples_per_second
        super(AudioSampleRate, self).__init__(self.one_sample, self.one_sample)

    def __int__(self):
        return self.samples_per_second

    def half_lapped(self):
        return SampleRate(
            self.one_sample * self.suggested_hop,
            self.one_sample * self.suggested_window)

    def windowing_scheme(self, duration_samples, frequency_samples=None):
        frequency_samples = frequency_samples or duration_samples
        return SampleRate(
            self.frequency * frequency_samples,
            self.duration * duration_samples)


class SR96000(AudioSampleRate):
    """
    A :class:`SampleRate` representing the common audio sampling rate 96kHz

    Examples:
        >>> from zounds import SR96000
        >>> sr = SR96000()
        >>> sr.samples_per_second
        96000
        >>> int(sr)
        96000
        >>> sr.nyquist
        48000
    """

    def __init__(self):
        super(SR96000, self).__init__(96000, 4096, 2048)


class SR48000(AudioSampleRate):
    """
    A :class:`SampleRate` representing the common audio sampling rate 48kHz

    Examples:
        >>> from zounds import SR48000
        >>> sr = SR48000()
        >>> sr.samples_per_second
        48000
        >>> int(sr)
        48000
        >>> sr.nyquist
        24000
    """

    def __init__(self):
        super(SR48000, self).__init__(48000, 2048, 1024)


class SR44100(AudioSampleRate):
    """
    A :class:`SampleRate` representing the common audio sampling rate 44.1kHz

    Examples:
        >>> from zounds import SR44100
        >>> sr = SR44100()
        >>> sr.samples_per_second
        44100
        >>> int(sr)
        44100
        >>> sr.nyquist
        22050
    """

    def __init__(self):
        super(SR44100, self).__init__(44100, 2048, 1024)


class SR22050(AudioSampleRate):
    """
    A :class:`SampleRate` representing the common audio sampling rate 22.025kHz

    Examples:
        >>> from zounds import SR22050
        >>> sr = SR22050()
        >>> sr.samples_per_second
        22050
        >>> int(sr)
        22050
        >>> sr.nyquist
        11025
    """

    def __init__(self):
        super(SR22050, self).__init__(22050, 1024, 512)


class SR16000(AudioSampleRate):
    """
        A :class:`SampleRate` representing the common audio sampling rate 16kHz

        Examples:
            >>> from zounds import SR16000
            >>> sr = SR16000()
            >>> sr.samples_per_second
            16000
            >>> int(sr)
            16000
            >>> sr.nyquist
            8000
        """

    def __init__(self):
        super(SR16000, self).__init__(16000, 512, 256)


class SR11025(AudioSampleRate):
    """
    A :class:`SampleRate` representing the common audio sampling rate 11.025kHz

    Examples:
        >>> from zounds import SR11025
        >>> sr = SR11025()
        >>> sr.samples_per_second
        11025
        >>> int(sr)
        11025
        >>> sr.nyquist
        5512
    """

    def __init__(self):
        super(SR11025, self).__init__(11025, 512, 256)


_samplerates = (
    SR96000(), SR48000(), SR44100(), SR22050(), SR16000(), SR11025())


def audio_sample_rate(samples_per_second):
    for sr in _samplerates:
        if samples_per_second == sr.samples_per_second:
            return sr
    raise ValueError(
        '{samples_per_second} is an invalid sample rate'.format(**locals()))


def nearest_audio_sample_rate(samples_per_second):
    samplerates = np.array([s.samples_per_second for s in _samplerates])
    diffs = np.abs(samples_per_second - samplerates)
    return _samplerates[np.argmin(diffs)]


class HalfLapped(SampleRate):
    def __init__(self, window_at_44100=2048, hop_at_44100=1024):
        one_sample_at_44100 = Picoseconds(int(1e12)) / 44100.
        window = one_sample_at_44100 * window_at_44100
        step = one_sample_at_44100 * hop_at_44100
        super(HalfLapped, self).__init__(step, window)
