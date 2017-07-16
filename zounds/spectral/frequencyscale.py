from __future__ import division
import numpy as np
import bisect


# TODO: What commonalities can be factored out of this class and TimeSlice?
# TODO: Deprecate stuff in psychoacoustics.py in favor of these classes
class FrequencyBand(object):
    """
    Represents an interval, or band of frequencies in hertz (cycles per second)
    """

    def __init__(self, start_hz, stop_hz):
        """
        :param start_hz: The lower bound for the frequency band, in hertz
        :param stop_hz: The upper bound for the frequency band, in hertz
        :return: a new FrequencyBand instance
        """
        super(FrequencyBand, self).__init__()
        if stop_hz <= start_hz:
            raise ValueError('stop_hz must be greater than start_hz')
        self.stop_hz = stop_hz
        self.start_hz = start_hz

    def __eq__(self, other):
        try:
            return \
                self.start_hz == other.start_hz \
                and self.stop_hz == other.stop_hz
        except AttributeError:
            return super(FrequencyBand, self).__eq__(other)

    @staticmethod
    def from_start(start_hz, bandwidth_hz):
        return FrequencyBand(start_hz, start_hz + bandwidth_hz)

    @staticmethod
    def from_center(center_hz, bandwidth_hz):
        half_bandwidth = bandwidth_hz / 2
        return FrequencyBand(
            center_hz - half_bandwidth, center_hz + half_bandwidth)

    @property
    def bandwidth(self):
        return self.stop_hz - self.start_hz

    @property
    def center_frequency(self):
        return self.start_hz + (self.bandwidth / 2)

    def __repr__(self):
        return '''FrequencyBand(
start_hz={start_hz},
stop_hz={stop_hz},
center={center},
bandwidth={bandwidth})'''.format(
            start_hz=self.start_hz,
            stop_hz=self.stop_hz,
            center=self.center_frequency,
            bandwidth=self.bandwidth)


class FrequencyScale(object):
    """
    Represents a set of frequency bands with monotonically increasing start
    frequencies
    """

    def __init__(self, frequency_band, n_bands, always_even=False):
        """
        :param frequency_band: A wide band that defines the boundaries for the
        entire scale.  E.g., one might want to generate a scale spanning the
        full range of (normal) human hearing by starting with
        FrequencyBand(20, 20000)
        :param n_bands: The total number of individual bands in the scale
        :return: a new FrequencyScale instance
        """
        super(FrequencyScale, self).__init__()
        self.always_even = always_even
        self.n_bands = n_bands
        self.frequency_band = frequency_band
        self._bands = None

    @property
    def bands(self):
        if self._bands is None:
            self._bands = self._compute_bands()
        return self._bands

    def _compute_bands(self):
        raise NotImplementedError()

    def __len__(self):
        return self.n_bands

    @property
    def center_frequencies(self):
        return (band.center_frequency for band in self)

    @property
    def bandwidths(self):
        return (band.bandwidth for band in self)

    @property
    def Q(self):
        """
        The quality factor of the scale, or, the ratio of center frequencies
        to bandwidths
        """
        return np.array(list(self.center_frequencies)) \
            / np.array(list(self.bandwidths))

    @property
    def start_hz(self):
        return self.frequency_band.start_hz

    @property
    def stop_hz(self):
        return self.frequency_band.stop_hz

    def get_slice(self, frequency_band):
        """
        Given a frequency band, and a frequency dimension comprised of n_samples,
        return a slice using integer indices that may be used to extract only
        the frequency samples that intersect with the frequency band
        :param frequency_band: The range of frequencies for which a slice should
        be produced
        :return: a slice
        """
        starts = [b.start_hz for b in self.bands]
        stops = [b.stop_hz for b in self.bands]
        start_index = bisect.bisect_left(stops, frequency_band.start_hz)
        stop_index = bisect.bisect_left(starts, frequency_band.stop_hz)

        if self.always_even and (stop_index - start_index) % 2:
            # KLUDGE: This is simple, but it may make sense to choose move the
            # upper *or* lower bound, based on which one introduces a lower
            # error
            stop_index += 1
        return slice(start_index, stop_index)

    def __eq__(self, other):
        return \
            self.__class__ == other.__class__ \
            and self.frequency_band == other.frequency_band \
            and self.n_bands == other.n_bands

    def __iter__(self):
        return iter(self.bands)

    def __getitem__(self, index):
        try:
            # index is an integer or slice
            bands = self.bands[index]
        except TypeError:
            # index is a frequency band
            bands = self.bands[self.get_slice(index)]

        try:
            freq_band = FrequencyBand(bands[0].start_hz, bands[-1].stop_hz)
            return self.__class__(freq_band, len(bands))
        except TypeError:
            # we've already got an individual band
            return bands

    def __str__(self):
        cls = self.__class__.__name__
        return '{cls}(band={self.frequency_band}, n_bands={self.n_bands})' \
            .format(**locals())

    def __repr__(self):
        return self.__str__()


class LinearScale(FrequencyScale):
    """
    A linear frequency scale with constant bandwidth.  Appropriate for use
    with transforms whose coefficients also lie on a linear frequency scale,
    e.g. the FFT or DCT transforms.
    """

    def __init__(self, frequency_band, n_bands, always_even=False):
        super(LinearScale, self).__init__(frequency_band, n_bands, always_even)

    @staticmethod
    def from_sample_rate(sample_rate, n_bands, always_even=False):
        fb = FrequencyBand(0, sample_rate.nyquist)
        return LinearScale(fb, n_bands, always_even=always_even)

    def _compute_bands(self):
        freqs = np.linspace(
            self.start_hz, self.stop_hz, self.n_bands, endpoint=False)
        # constant, non-overlapping bandwidth
        bandwidth = freqs[1] - freqs[0]
        return tuple(FrequencyBand(f, f + bandwidth) for f in freqs)


class LogScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands, always_even=False):
        super(LogScale, self).__init__(
            frequency_band, n_bands, always_even=always_even)

    def _compute_bands(self):
        center_freqs = np.logspace(
            np.log10(self.start_hz),
            np.log10(self.stop_hz),
            self.n_bands + 1)
        # variable bandwidth
        bandwidths = np.diff(center_freqs)
        return tuple(FrequencyBand.from_center(cf, bw)
                     for (cf, bw) in zip(center_freqs[:-1], bandwidths))


class ConstantQScale(FrequencyScale):
    def __init__(self, lowest_center_freq_hz, n_octaves, n_bands_per_octave):
        self.__bands = []
        total_bands = n_octaves * n_bands_per_octave
        pos = None
        for i in xrange(total_bands):
            bandwidth = \
                ((2 ** (1 / n_bands_per_octave)) ** i) * lowest_center_freq_hz
            half_bandwidth = bandwidth / 2
            if pos is None:
                pos = lowest_center_freq_hz - half_bandwidth
            self.__bands.append(FrequencyBand(pos, pos + bandwidth))
            pos += bandwidth
        fb = FrequencyBand(self.__bands[0].start_hz, self.__bands[-1].stop_hz)
        super(ConstantQScale, self).__init__(fb, len(self.__bands))

    def _compute_bands(self):
        return self.__bands


class GeometricScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(GeometricScale, self).__init__(frequency_band, n_bands)

    def _compute_bands(self):
        start_freqs = np.geomspace(
            self.start_hz, self.stop_hz, num=self.n_bands + 1, endpoint=False)
        bandwidths = np.diff(start_freqs)
        return tuple(FrequencyBand.from_start(sf, bw)
                     for (sf, bw) in zip(start_freqs[:-1], bandwidths))


class BarkScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(BarkScale, self).__init__(frequency_band, n_bands)


class MelScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(MelScale, self).__init__(frequency_band, n_bands)
