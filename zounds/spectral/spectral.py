from __future__ import division

import numpy as np
from featureflow import Node
from scipy.fftpack import dct
from scipy.stats.mstats import gmean

from frequencyscale import LinearScale
from psychacoustics import Chroma as ChromaScale, Bark as BarkScale
from tfrepresentation import FrequencyDimension
from frequencyadaptive import FrequencyAdaptive
from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.nputil import safe_log
from zounds.timeseries import SR44100, audio_sample_rate


class FFT(Node):
    def __init__(self, needs=None, axis=-1):
        super(FFT, self).__init__(needs=needs)
        self._axis = axis

    def _process(self, data):
        transformed = np.fft.rfft(data, axis=self._axis, norm='ortho')

        sr = audio_sample_rate(
            int(data.shape[1] / data.dimensions[0].duration_in_seconds))
        scale = LinearScale.from_sample_rate(sr, transformed.shape[-1])

        yield ArrayWithUnits(
            transformed, [data.dimensions[0], FrequencyDimension(scale)])


class DCT(Node):
    """
    Type II Discrete Cosine Transform
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
    Type IV Discrete Cosine Transform.  This transform is its own inverse
    """

    def __init__(self, scale_always_even=False, needs=None):
        super(DCTIV, self).__init__(needs=needs)
        self.scale_always_even = scale_always_even

    def _process_raw(self, data):
        l = data.shape[1]
        tf = np.arange(0, l)
        z = np.zeros((len(data), l * 2))
        z[:, :l] = data * np.exp(-1j * np.pi * tf / 2 / l)
        z = np.fft.fft(z)[:, :l]
        raw = np.sqrt(2 / l) * \
              np.real(z * np.exp(-1j * np.pi * (tf + 0.5) / 2 / l))
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
    Modified Discrete Cosine Transform
    """

    def __init__(self, needs=None):
        super(MDCT, self).__init__(needs=needs)

    def _process_raw(self, data):
        l = data.shape[1] // 2
        t = np.arange(0, 2 * l)
        f = np.arange(0, l)
        cpi = -1j * np.pi
        a = data * np.exp(cpi * t / 2 / l)
        b = np.fft.fft(a)
        c = b[:, :l]
        transformed = np.sqrt(2 / l) * np.real(
            c * np.exp(cpi * (f + 0.5) * (l + 1) / 2 / l))
        return transformed

    def _process(self, data):
        transformed = self._process_raw(data)

        sr = audio_sample_rate(data.dimensions[1].samples_per_second)
        scale = LinearScale.from_sample_rate(sr, transformed.shape[1])

        yield ArrayWithUnits(
            transformed, [data.dimensions[0], FrequencyDimension(scale)])


class FrequencyAdaptiveTransform(Node):
    def __init__(
            self,
            transform=None,
            scale=None,
            window_func=None,
            needs=None):
        super(FrequencyAdaptiveTransform, self).__init__(needs=needs)
        self._window_func = window_func or np.ones
        self._scale = scale
        self._transform = transform

    def _process_band(self, data, band):
        raw_coeffs = data[:, band]
        window = self._window_func(raw_coeffs.shape[1])
        return self._transform(raw_coeffs * window[None, :], norm='ortho')

    def _process(self, data):
        yield FrequencyAdaptive(
            [self._process_band(data, band) for band in self._scale],
            data.dimensions[0],
            self._scale)


# TODO: This constructor should not take a samplerate; that information should
# be encapsulated in the data that's passed in
class Chroma(Node):
    def __init__(
            self,
            needs=None,
            samplerate=SR44100(),
            nbins=12,
            a440=440.):
        super(Chroma, self).__init__(needs=needs)
        self._nbins = nbins
        self._a440 = a440
        self._samplerate = samplerate
        self._chroma_scale = None

    def _process(self, data):
        data = np.abs(data)
        if self._chroma_scale is None:
            self._chroma_scale = ChromaScale( \
                self._samplerate.samples_per_second,
                data.shape[1] * 2,
                nbands=self._nbins)

        yield ArrayWithUnits(
            self._chroma_scale.transform(data),
            [data.dimensions[0], IdentityDimension()])


# TODO: This constructor should not take a samplerate; that information should
# be encapsulated in the data that's passed in
class BarkBands(Node):
    def __init__(
            self,
            needs=None,
            samplerate=SR44100(),
            n_bands=100,
            start_freq_hz=50,
            stop_freq_hz=2e4):
        super(BarkBands, self).__init__(needs=needs)

        self._samplerate = samplerate
        self._n_bands = n_bands
        self._start_freq_hz = start_freq_hz
        self._stop_freq_hz = stop_freq_hz
        self._bark_scale = None

    def _process(self, data):
        data = np.abs(data)
        if self._bark_scale is None:
            self._bark_scale = BarkScale(
                self._samplerate.samples_per_second,
                data.shape[1] * 2,
                self._n_bands,
                self._start_freq_hz,
                self._stop_freq_hz)

        yield ArrayWithUnits(
            self._bark_scale.transform(data),
            [data.dimensions[0], IdentityDimension()])


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
