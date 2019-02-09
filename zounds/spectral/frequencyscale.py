from __future__ import division
import numpy as np
import bisect


class Hertz(float):
    def __init__(self, hz):
        super(Hertz, self).__init__(hz)
        self.hz = hz

    def __neg__(self):
        return Hertz(-self.hz)

    def __add__(self, other):
        return Hertz(self.hz + other.hz)

    
class Hz(Hertz):
    pass


# TODO: What commonalities can be factored out of this class and TimeSlice?
class FrequencyBand(object):
    """
    Represents an interval, or band of frequencies in hertz (cycles per second)

    Args:
        start_hz (float): The lower bound of the frequency band in hertz
        stop_hz (float): The upper bound of the frequency band in hertz

    Examples::
        >>> import zounds
        >>> band = zounds.FrequencyBand(500, 1000)
        >>> band.center_frequency
        750.0
        >>> band.bandwidth
        500
    """

    def __init__(self, start_hz, stop_hz):
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
        """
        Return the intersection between this frequency band and another.

        Args:
            other (FrequencyBand): the instance to intersect with

        Examples::
            >>> import zounds
            >>> b1 = zounds.FrequencyBand(500, 1000)
            >>> b2 = zounds.FrequencyBand(900, 2000)
            >>> intersection = b1.intersect(b2)
            >>> intersection.start_hz, intersection.stop_hz
            (900, 1000)
        """
        lowest_stop = min(self.stop_hz, other.stop_hz)
        highest_start = max(self.start_hz, other.start_hz)
        return FrequencyBand(highest_start, lowest_stop)

    @classmethod
    def audible_range(cls, samplerate):
        return FrequencyBand(Hz(20), Hz(samplerate.nyquist))

    def bandwidth_ratio(self, other):
        return other.bandwidth / self.bandwidth

    def intersection_ratio(self, other):
        intersection = self.intersect(other)
        return self.bandwidth_ratio(intersection)

    @staticmethod
    def from_start(start_hz, bandwidth_hz):
        """
        Produce a :class:`FrequencyBand` instance from a lower bound and
        bandwidth

        Args:
            start_hz (float): the lower bound of the desired FrequencyBand
            bandwidth_hz (float): the bandwidth of the desired FrequencyBand

        """
        return FrequencyBand(start_hz, start_hz + bandwidth_hz)

    @staticmethod
    def from_center(center_hz, bandwidth_hz):
        half_bandwidth = bandwidth_hz / 2
        return FrequencyBand(
            center_hz - half_bandwidth, center_hz + half_bandwidth)

    @property
    def bandwidth(self):
        """
        The span of this frequency band, in hertz
        """
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

    Args:
        frequency_band (FrequencyBand): A band representing the entire span of
            this scale.  E.g., one might want to generate a scale spanning the
            entire range of human hearing by starting with
            :code:`FrequencyBand(20, 20000)`
        n_bands (int): The number of bands in this scale
        always_even (bool): when converting frequency slices to integer indices
            that numpy can understand, should the slice size always be even?

    See Also:
        :class:`~zounds.spectral.LinearScale`
        :class:`~zounds.spectral.GeometricScale`
    """

    def __init__(self, frequency_band, n_bands, always_even=False):
        super(FrequencyScale, self).__init__()
        self.always_even = always_even
        self.n_bands = n_bands
        self.frequency_band = frequency_band
        self._bands = None
        self._starts = None
        self._stops = None

    @property
    def bands(self):
        """
        An iterable of all bands in this scale
        """
        if self._bands is None:
            self._bands = self._compute_bands()
        return self._bands

    @property
    def band_starts(self):
        if self._starts is None:
            self._starts = [b.start_hz for b in self.bands]
        return self._starts

    @property
    def band_stops(self):
        if self._stops is None:
            self._stops = [b.stop_hz for b in self.bands]
        return self._stops

    def _compute_bands(self):
        raise NotImplementedError()

    def __len__(self):
        return self.n_bands

    @property
    def center_frequencies(self):
        """
        An iterable of the center frequencies of each band in this scale
        """
        return (band.center_frequency for band in self)

    @property
    def bandwidths(self):
        """
        An iterable of the bandwidths of each band in this scale
        """
        return (band.bandwidth for band in self)

    def ensure_overlap_ratio(self, required_ratio=0.5):
        """
        Ensure that every adjacent pair of frequency bands meets the overlap
        ratio criteria.  This can be helpful in scenarios where a scale is
        being used in an invertible transform, and something like the `constant
        overlap add constraint
        <https://ccrma.stanford.edu/~jos/sasp/Constant_Overlap_Add_COLA_Cases.html>`_
        must be met in order to not introduce artifacts in the reconstruction.

        Args:
            required_ratio (float): The required overlap ratio between all
                adjacent frequency band pairs

        Raises:
            AssertionError: when the overlap ratio for one or more adjacent
                frequency band pairs is not met
        """

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
        """
        The lower bound of this frequency scale
        """
        return self.frequency_band.start_hz

    @property
    def stop_hz(self):
        """
        The upper bound of this frequency scale
        """
        return self.frequency_band.stop_hz

    def _basis(self, other_scale, window):
        weights = np.zeros((len(self), len(other_scale)))
        for i, band in enumerate(self):
            band_slice = other_scale.get_slice(band)
            slce = weights[i, band_slice]
            slce[:] = window * np.ones(len(slce))
        return weights

    def apply(self, time_frequency_repr, window):
        basis = self._basis(time_frequency_repr.dimensions[-1].scale, window)
        transformed = np.dot(basis, time_frequency_repr.T).T
        return transformed

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

    def get_slice(self, frequency_band):
        """
        Given a frequency band, and a frequency dimension comprised of
        n_samples, return a slice using integer indices that may be used to
        extract only the frequency samples that intersect with the frequency
        band
        """
        index = frequency_band

        if isinstance(index, slice):
            types = {
                index.start.__class__,
                index.stop.__class__,
                index.step.__class__
            }

            if Hertz not in types:
                return index

            try:
                start = Hertz(0) if index.start is None else index.start
                if start < Hertz(0):
                    start = self.stop_hz + start
                stop = self.stop_hz if index.stop is None else index.stop
                if stop < Hertz(0):
                    stop = self.stop_hz + stop
                frequency_band = FrequencyBand(start, stop)
            except (ValueError, TypeError):
                pass

        start_index = bisect.bisect_left(
            self.band_stops, frequency_band.start_hz)
        stop_index = bisect.bisect_left(
            self.band_starts, frequency_band.stop_hz)

        if self.always_even and (stop_index - start_index) % 2:
            # KLUDGE: This is simple, but it may make sense to choose move the
            # upper *or* lower bound, based on which one introduces a lower
            # error
            stop_index += 1
        return slice(start_index, stop_index)

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

    Args:
        frequency_band (FrequencyBand): A band representing the entire span of
            this scale.  E.g., one might want to generate a scale spanning the
            entire range of human hearing by starting with
            :code:`FrequencyBand(20, 20000)`
        n_bands (int): The number of bands in this scale
        always_even (bool): when converting frequency slices to integer indices
            that numpy can understand, should the slice size always be even?

    Examples:
        >>> from zounds import FrequencyBand, LinearScale
        >>> scale = LinearScale(FrequencyBand(20, 20000), 10)
        >>> scale
        LinearScale(band=FrequencyBand(
        start_hz=20,
        stop_hz=20000,
        center=10010.0,
        bandwidth=19980), n_bands=10)
        >>> scale.Q
        array([ 0.51001001,  1.51001001,  2.51001001,  3.51001001,  4.51001001,
                5.51001001,  6.51001001,  7.51001001,  8.51001001,  9.51001001])
    """

    def __init__(self, frequency_band, n_bands, always_even=False):
        super(LinearScale, self).__init__(frequency_band, n_bands, always_even)

    @staticmethod
    def from_sample_rate(sample_rate, n_bands, always_even=False):
        """
        Return a :class:`~zounds.spectral.LinearScale` instance whose upper
        frequency bound is informed by the nyquist frequency of the sample rate.

        Args:
            sample_rate (SamplingRate): the sample rate whose nyquist frequency
                will serve as the upper frequency bound of this scale
            n_bands (int): the number of evenly-spaced frequency bands
        """
        fb = FrequencyBand(0, sample_rate.nyquist)
        return LinearScale(fb, n_bands, always_even=always_even)

    def _compute_bands(self):
        freqs = np.linspace(
            self.start_hz, self.stop_hz, self.n_bands, endpoint=False)
        # constant, non-overlapping bandwidth
        bandwidth = freqs[1] - freqs[0]
        return tuple(FrequencyBand(f, f + bandwidth) for f in freqs)


# class LogScale(FrequencyScale):
#     def __init__(self, frequency_band, n_bands, always_even=False):
#         super(LogScale, self).__init__(
#             frequency_band, n_bands, always_even=always_even)
#
#     def _compute_bands(self):
#         center_freqs = np.logspace(
#             np.log10(self.start_hz),
#             np.log10(self.stop_hz),
#             self.n_bands + 1)
#         # variable bandwidth
#         bandwidths = np.diff(center_freqs)
#         return tuple(FrequencyBand.from_center(cf, bw)
#                      for (cf, bw) in zip(center_freqs[:-1], bandwidths))


class GeometricScale(FrequencyScale):
    """
    A constant-Q scale whose center frequencies progress geometrically rather
    than linearly

    Args:
        start_center_hz (int): the center frequency of the first band in the
            scale
        stop_center_hz (int): the center frequency of the last band in the scale
        bandwidth_ratio (float): the center frequency to bandwidth ratio
        n_bands (int): the total number of bands

    Examples:
        >>> from zounds import GeometricScale
        >>> scale = GeometricScale(20, 20000, 0.05, 10)
        >>> scale
        GeometricScale(band=FrequencyBand(
        start_hz=19.5,
        stop_hz=20500.0,
        center=10259.75,
        bandwidth=20480.5), n_bands=10)
        >>> scale.Q
        array([ 20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.])
        >>> list(scale.center_frequencies)
        [20.000000000000004, 43.088693800637671, 92.831776672255558,
            200.00000000000003, 430.88693800637651, 928.31776672255558,
            2000.0000000000005, 4308.8693800637648, 9283.1776672255564,
            20000.000000000004]
    """

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
    """
    A scale where the frequency bands are provided explicitly, rather than
    computed

    Args:
        bands (list of FrequencyBand): The explicit bands used by this scale

    See Also:
        :class:`~zounds.spectral.FrequencyAdaptive`
    """

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


class Bark(Hertz):
    def __init__(self, bark):
        self.bark = bark
        super(Bark, self).__init__(Bark.to_hz(bark))

    @staticmethod
    def to_hz(bark):
        return 300. * ((np.e ** (bark / 6.0)) - (np.e ** (-bark / 6.)))

    @staticmethod
    def to_bark(hz):
        return 6. * np.log((hz / 600.) + np.sqrt((hz / 600.) ** 2 + 1))


def equivalent_rectangular_bandwidth(hz):
    return (0.108 * hz) + 24.7


class BarkScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(BarkScale, self).__init__(frequency_band, n_bands)

    def _compute_bands(self):
        start = Bark.to_bark(self.frequency_band.start_hz)
        stop = Bark.to_bark(self.frequency_band.stop_hz)
        barks = np.linspace(start, stop, self.n_bands)
        center_frequencies_hz = Bark.to_hz(barks)
        bandwidths = equivalent_rectangular_bandwidth(center_frequencies_hz)
        return [
            FrequencyBand.from_center(c, b)
            for c, b in zip(center_frequencies_hz, bandwidths)]


class Mel(Hertz):
    def __init__(self, mel):
        self.mel = mel
        super(Mel, self).__init__(Mel.to_hz(mel))

    @staticmethod
    def to_hz(mel):
        return 700 * ((np.e ** (mel / 1127)) - 1)

    @staticmethod
    def to_mel(hz):
        return 1127 * np.log(1 + (hz / 700))


class MelScale(FrequencyScale):
    def __init__(self, frequency_band, n_bands):
        super(MelScale, self).__init__(frequency_band, n_bands)

    def _compute_bands(self):
        start = Mel.to_mel(self.frequency_band.start_hz)
        stop = Mel.to_mel(self.frequency_band.stop_hz)
        mels = np.linspace(start, stop, self.n_bands)
        center_frequencies_hz = Mel.to_hz(mels)
        bandwidths = equivalent_rectangular_bandwidth(center_frequencies_hz)
        return [
            FrequencyBand.from_center(c, b)
            for c, b in zip(center_frequencies_hz, bandwidths)]


class ChromaScale(FrequencyScale):
    def __init__(self, frequency_band):
        self._a440 = 440.
        self._a = 2 ** (1 / 12.)
        super(ChromaScale, self).__init__(frequency_band, n_bands=12)

    def _compute_bands(self):
        raise NotImplementedError()

    def get_slice(self, frequency_band):
        raise NotImplementedError()

    def _semitones_to_hz(self, semitone):
        return self._a440 * (self._a ** semitone)

    def _hz_to_semitones(self, hz):
        """
        Convert hertz into a number of semitones above or below some reference
        value, in this case, A440
        """
        return np.log(hz / self._a440) / np.log(self._a)

    def _basis(self, other_scale, window):
        basis = np.zeros((self.n_bands, len(other_scale)))

        # for each tone in the twelve-tone scale, generate narrow frequency
        # bands for every octave of that note that falls within the frequency
        # band.
        start_semitones = \
            int(np.round(self._hz_to_semitones(self.frequency_band.start_hz)))
        stop_semitones = \
            int(np.round(self._hz_to_semitones(self.frequency_band.stop_hz)))

        semitones = np.arange(start_semitones - 1, stop_semitones)
        hz = self._semitones_to_hz(semitones)

        bands = []
        for i in xrange(0, len(semitones) - 2):
            fh, mh, lh = hz[i: i + 3]
            bands.append(FrequencyBand(fh, lh))

        for semitone, band in zip(semitones, bands):
            slce = other_scale.get_slice(band)
            chroma_index = semitone % self.n_bands
            slce = basis[chroma_index, slce]
            slce[:] += np.ones(len(slce)) * window

        return basis
