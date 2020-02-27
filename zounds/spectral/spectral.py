
import numpy as np
from featureflow import Node
from scipy.fftpack import dct
from scipy.stats.mstats import gmean

from .functional import fft, mdct
from .frequencyscale import LinearScale, ChromaScale, BarkScale
from .weighting import AWeighting
from .tfrepresentation import FrequencyDimension
from .frequencyadaptive import FrequencyAdaptive
from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.nputil import safe_log
from zounds.timeseries import audio_sample_rate
from .sliding_window import HanningWindowingFunc


class FrequencyWeighting(Node):
    """
    `FrequencyWeighting` is a processing node that expects to be passed an
    :class:`~zounds.core.ArrayWithUnits` instance whose last dimension is a
    :class:`~zounds.spectral.FrequencyDimension`

    Args:
        weighting (FrequencyWeighting): the frequency weighting to apply
        needs (Node): a processing node on which this node depends whose last
            dimension is a :class:`~zounds.spectral.FrequencyDimension`
    """

    def __init__(self, weighting=None, needs=None):
        super(FrequencyWeighting, self).__init__(needs=needs)
        self.weighting = weighting

    def _process(self, data):
        yield data * self.weighting


class FFT(Node):
    """
    A processing node that performs an FFT of a real-valued signal

    Args:
        axis (int): The axis over which the FFT should be computed
        padding_samples (int): number of zero samples to pad each window with
            before applying the FFT
        needs (Node): a processing node on which this one depends

    See Also:
        :class:`~zounds.synthesize.FFTSynthesizer`
    """

    def __init__(self, needs=None, axis=-1, padding_samples=0):
        super(FFT, self).__init__(needs=needs)
        self._axis = axis
        self._padding_samples = padding_samples

    def _process(self, data):
        yield fft(data, axis=self._axis, padding_samples=self._padding_samples)


class DCT(Node):
    """
    A processing node that performs a Type II Discrete Cosine Transform
    (https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II) of the
    input

    Args:
        axis (int): The axis over which to perform the DCT transform
        needs (Node): a processing node on which this one depends

    See Also:
        :class:`~zounds.synthesize.DctSynthesizer`
    """

    def __init__(self, axis=-1, scale_always_even=False, needs=None):
        super(DCT, self).__init__(needs=needs)
        self.scale_always_even = scale_always_even
        self._axis = axis

    def _process(self, data):
        transformed = dct(data, norm='ortho', axis=self._axis)

        sr = audio_sample_rate(
            int(data.shape[1] / data.dimensions[0].duration_in_seconds))
        scale = LinearScale.from_sample_rate(
            sr, transformed.shape[-1], always_even=self.scale_always_even)

        yield ArrayWithUnits(
            transformed, [data.dimensions[0], FrequencyDimension(scale)])


class DCTIV(Node):
    """
    A processing node that performs a Type IV Discrete Cosine Transform
    (https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-IV) of the
    input

    Args:
        needs (Node): a processing node on which this one depends

    See Also:
        :class:`~zounds.synthesize.DCTIVSynthesizer`
    """

    def __init__(self, scale_always_even=False, needs=None):
        super(DCTIV, self).__init__(needs=needs)
        self.scale_always_even = scale_always_even

    def _process_raw(self, data):
        l = data.shape[1]
        tf = np.arange(0, l)
        z = np.zeros((len(data), l * 2))
        z[:, :l] = (data * np.exp(-1j * np.pi * tf / 2 / l)).real
        z = np.fft.fft(z)[:, :l]
        raw = np.sqrt(2 / l) * \
              (z * np.exp(-1j * np.pi * (tf + 0.5) / 2 / l)).real
        return raw

    def _process(self, data):
        raw = self._process_raw(data)
        sr = audio_sample_rate(
            int(data.shape[1] / data.dimensions[0].duration_in_seconds))
        scale = LinearScale.from_sample_rate(
            sr, data.shape[1], always_even=self.scale_always_even)
        yield ArrayWithUnits(
            raw, [data.dimensions[0], FrequencyDimension(scale)])


class MDCT(Node):
    """
    A processing node that performs a modified discrete cosine transform
    (https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform) of the
    input.

    This is really just a lapped version of the DCT-IV transform

    Args:
        needs (Node): a processing node on which this one depends

    See Also:
        :class:`~zounds.synthesize.MDCTSynthesizer`
    """

    def __init__(self, needs=None):
        super(MDCT, self).__init__(needs=needs)

    def _process(self, data):
        transformed = mdct(data)

        sr = audio_sample_rate(data.dimensions[1].samples_per_second)
        scale = LinearScale.from_sample_rate(sr, transformed.shape[1])

        yield ArrayWithUnits(
            transformed, [data.dimensions[0], FrequencyDimension(scale)])


class FrequencyAdaptiveTransform(Node):
    """
    A processing node that expects to receive the input from a frequency domain
    transformation (e.g. :class:`~zounds.spectral.FFT`), and produces a
    :class:`~zounds.spectral.FrequencyAdaptive` instance where time resolution
    can vary by frequency.  This is similar to, but not precisely the same as
    ideas introduced in:

    * `A quasi-orthogonal, invertible, and perceptually relevant time-frequency transform for audio coding <https://hal-amu.archives-ouvertes.fr/hal-01194806/document>`_
    * `A FRAMEWORK FOR INVERTIBLE, REAL-TIME CONSTANT-Q TRANSFORMS <http://www.univie.ac.at/nonstatgab/pdf_files/dogrhove12_amsart.pdf>`_

    Args:
        transform (function): the transform to be applied to each frequency band
        scale (FrequencyScale): the scale used to take frequency band slices
        window_func (numpy.ndarray): the windowing function to apply each band
            before the transform is applied
        check_scale_overlap_ratio (bool): If this feature is to be used for
            resynthesis later, ensure that each frequency band overlaps with
            the previous one by at least half, to ensure artifact-free synthesis

    See Also:
        :class:`~zounds.spectral.FrequencyAdaptive`
        :class:`~zounds.synthesize.FrequencyAdaptiveDCTSynthesizer`
        :class:`~zounds.synthesize.FrequencyAdaptiveFFTSynthesizer`
    """

    def __init__(
            self,
            transform=None,
            scale=None,
            window_func=None,
            check_scale_overlap_ratio=False,
            needs=None):
        super(FrequencyAdaptiveTransform, self).__init__(needs=needs)

        if check_scale_overlap_ratio:
            try:
                scale.ensure_overlap_ratio(0.5)
            except AssertionError as e:
                raise ValueError(*e.args)

        self._window_func = window_func or np.ones
        self._scale = scale
        self._transform = transform

    def _process_band(self, data, band):
        try:
            raw_coeffs = data[:, band]
        except IndexError:
            raise ValueError(
                'data must have FrequencyDimension as its last dimension, '
                'but it was {dim}'.format(dim=data.dimensions[-1]))
        window = self._window_func(raw_coeffs.shape[1])
        return self._transform(raw_coeffs * window[None, :], norm='ortho')

    def _process(self, data):
        yield FrequencyAdaptive(
            [self._process_band(data, band) for band in self._scale],
            data.dimensions[0],
            self._scale)


class BaseScaleApplication(Node):
    def __init__(self, scale, window, needs=None):
        super(BaseScaleApplication, self).__init__(needs=needs)
        self.window = window
        self.scale = scale

    def _new_dim(self):
        return FrequencyDimension(self.scale)

    def _preprocess(self, data):
        return data

    def _process(self, data):
        x = self._preprocess(data)
        x = self.scale.apply(x, self.window)
        yield ArrayWithUnits(
            x, data.dimensions[:-1] + (self._new_dim(),))


class Chroma(BaseScaleApplication):
    def __init__(
            self, frequency_band, window=HanningWindowingFunc(), needs=None):
        super(Chroma, self).__init__(
            ChromaScale(frequency_band), window, needs=needs)

    def _new_dim(self):
        return IdentityDimension()

    def _preprocess(self, data):
        return np.abs(data) * AWeighting()


class BarkBands(BaseScaleApplication):
    def __init__(
            self,
            frequency_band,
            n_bands=100,
            window=HanningWindowingFunc(),
            needs=None):
        super(BarkBands, self).__init__(
            BarkScale(frequency_band, n_bands), window, needs=needs)

    def _preprocess(self, data):
        return np.abs(data)


class SpectralCentroid(Node):
    """
    Indicates where the "center of mass" of the spectrum is. Perceptually,
    it has a robust connection with the impression of "brightness" of a
    sound.  It is calculated as the weighted mean of the frequencies
    present in the signal, determined using a Fourier transform, with
    their magnitudes as the weights...

    -- http://en.wikipedia.org/wiki/Spectral_centroid
    """

    def __init__(self, needs=None):
        super(SpectralCentroid, self).__init__(needs=needs)

    def _first_chunk(self, data):
        self._bins = np.arange(1, data.shape[-1] + 1)
        self._bins_sum = np.sum(self._bins)
        return data

    def _process(self, data):
        data = np.abs(data)
        yield (data * self._bins).sum(axis=1) / self._bins_sum


class SpectralFlatness(Node):
    """
    Spectral flatness or tonality coefficient, also known as Wiener
    entropy, is a measure used in digital signal processing to characterize an
    audio spectrum. Spectral flatness is typically measured in decibels, and
    provides a way to quantify how tone-like a sound is, as opposed to being
    noise-like. The meaning of tonal in this context is in the sense of the
    amount of peaks or resonant structure in a power spectrum, as opposed to
    flat spectrum of a white noise. A high spectral flatness indicates that
    the spectrum has a similar amount of power in all spectral bands - this
    would sound similar to white noise, and the graph of the spectrum would
    appear relatively flat and smooth. A low spectral flatness indicates that
    the spectral power is concentrated in a relatively small number of
    bands - this would typically sound like a mixture of sine waves, and the
    spectrum would appear "spiky"...

    -- http://en.wikipedia.org/wiki/Spectral_flatness
    """

    def __init__(self, needs=None):
        super(SpectralFlatness, self).__init__(needs=needs)

    def _process(self, data):
        data = np.abs(data)
        mean = data.mean(axis=1)
        mean[mean == 0] = -1e5
        flatness = gmean(data, axis=1) / mean
        yield ArrayWithUnits(flatness, data.dimensions[:1])


class BFCC(Node):
    """
    Bark frequency cepstral coefficients
    """

    def __init__(self, needs=None, n_coeffs=13, exclude=1):
        super(BFCC, self).__init__(needs=needs)
        self._n_coeffs = n_coeffs
        self._exclude = exclude

    def _process(self, data):
        data = np.abs(data)
        bfcc = dct(safe_log(data), axis=1) \
            [:, self._exclude: self._exclude + self._n_coeffs]

        yield ArrayWithUnits(
            bfcc.copy(), [data.dimensions[0], IdentityDimension()])
