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

    def __init__(self, frequency_band, n_bands):
        """
        :param frequency_band: A wide band that defines the boundaries for the
        entire scale.  E.g., one might want to generate a scale spanning the
        full range of (normal) human hearing by starting with
        FrequencyBand(20, 20000)
        :param n_bands: The total number of individual bands in the scale
        :return: a new FrequencyScale instance
        """
        super(FrequencyScale, self).__init__()
        self.n_bands = n_bands
        self.frequency_band = frequency_band

    def __len__(self):
        return self.n_bands

    @property
    def center_frequencies(self):
        return (band.center_frequency for band in self)

    @property
    def bandwidths(self):
        return (band.bandwidth for band in self)

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
        bands = list(self)
        starts = [b.start_hz for b in bands]
        stops = [b.stop_hz for b in bands]
        start_index = bisect.bisect_left(stops, frequency_band.start_hz)
        stop_index = bisect.bisect_left(starts, frequency_band.stop_hz)
        return slice(start_index, stop_index)

    def __iter__(self):
        raise NotImplementedError()


class LinearScale(FrequencyScale):
    """
    A linear frequency scale with constant bandwidth.  Appropriate for use
    with transforms whose coefficients also lie on a linear frequency scale,
    e.g. the FFT or DCT transforms.
    """

    def __init__(self, frequency_band, n_bands):
        super(LinearScale, self).__init__(frequency_band, n_bands)

    @staticmethod
    def from_sample_rate(sample_rate, n_bands):
        fb = FrequencyBand(0, sample_rate.nyquist)
        return LinearScale(fb, n_bands)

    def __iter__(self):
        freqs = np.linspace(self.start_hz, self.stop_hz, self.n_bands)
        # constant, non-overlapping bandwidth
        bandwidth = freqs[1] - freqs[0]
        return (FrequencyBand(f, f + bandwidth) for f in freqs)


class LogScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(LogScale, self).__init__(frequency_band, n_bands)

    def __iter__(self):
        center_freqs = np.logspace(
                np.log10(self.start_hz),
                np.log10(self.stop_hz),
                self.n_bands + 1)
        bandwidths = np.diff(center_freqs)
        return (FrequencyBand.from_center(cf, bw)
                for (cf, bw) in zip(center_freqs[:-1], bandwidths))


class BarkScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(BarkScale, self).__init__(frequency_band, n_bands)


class MelScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(MelScale, self).__init__(frequency_band, n_bands)
