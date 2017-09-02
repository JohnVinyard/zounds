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

    def __hash__(self):
        return (self.__class__.__name__, self.start_hz, self.stop_hz).__hash__()

    def intersect(self, other):
        lowest_stop = min(self.stop_hz, other.stop_hz)
        highest_start = max(self.start_hz, other.start_hz)
        return FrequencyBand(highest_start, lowest_stop)

    def bandwidth_ratio(self, other):
        return other.bandwidth / self.bandwidth

    def intersection_ratio(self, other):
        intersection = self.intersect(other)
        return self.bandwidth_ratio(intersection)

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

    def ensure_overlap_ratio(self, required_ratio=0.5):
        msg = \
            'band {i}: ratio must be at least {required_ratio} but was {ratio}'

        for i in xrange(0, len(self) - 1):
            b1 = self[i]
            b2 = self[i + 1]

            try:
                ratio = b1.intersection_ratio(b2)
            except ValueError:
                ratio = 0

            if ratio < required_ratio:
                raise AssertionError(msg.format(**locals()))

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

    def _construct_scale_from_slice(self, bands):
        freq_band = FrequencyBand(bands[0].start_hz, bands[-1].stop_hz)
        return self.__class__(freq_band, len(bands))

    def __getitem__(self, index):
        try:
            # index is an integer or slice
            bands = self.bands[index]
        except TypeError:
            # index is a frequency band
            bands = self.bands[self.get_slice(index)]

        if isinstance(bands, FrequencyBand):
            return bands

        return self._construct_scale_from_slice(bands)

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


class GeometricScale(FrequencyScale):
    def __init__(
            self,
            start_center_hz,
            stop_center_hz,
            bandwidth_ratio,
            n_bands,
            always_even=False):
        self.__bands = [
            FrequencyBand.from_center(cf, cf * bandwidth_ratio)
            for cf in np.geomspace(start_center_hz, stop_center_hz, num=n_bands)
            ]
        band = FrequencyBand(self.__bands[0].start_hz, self.__bands[-1].stop_hz)
        super(GeometricScale, self).__init__(
            band, n_bands, always_even=always_even)
        self.start_center_hz = start_center_hz
        self.stop_center_hz = stop_center_hz
        self.bandwidth_ratio = bandwidth_ratio

    def _construct_scale_from_slice(self, bands):
        return ExplicitScale(bands)

    def __eq__(self, other):
        return \
            super(GeometricScale, self).__eq__(other) \
            and self.start_center_hz == other.start_center_hz \
            and self.stop_center_hz == other.stop_center_hz \
            and self.bandwidth_ratio == other.bandwidth_ratio

    def _compute_bands(self):
        return self.__bands


class ExplicitScale(FrequencyScale):
    def __init__(self, bands):
        bands = list(bands)
        frequency_band = FrequencyBand(bands[0].start_hz, bands[-1].stop_hz)
        super(ExplicitScale, self).__init__(
            frequency_band, len(bands), always_even=False)
        self._bands = bands

    def _construct_scale_from_slice(self, bands):
        return ExplicitScale(bands)

    def _compute_bands(self):
        return self._bands

    def __eq__(self, other):
        return all([a == b for (a, b) in zip(self, other)])


class BarkScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(BarkScale, self).__init__(frequency_band, n_bands)


class MelScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(MelScale, self).__init__(frequency_band, n_bands)


class ChromaScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(ChromaScale, self).__init__(frequency_band, n_bands)
