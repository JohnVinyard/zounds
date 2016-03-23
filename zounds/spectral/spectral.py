from featureflow import Node
import numpy as np
from scipy.fftpack import dct
from scipy.stats.mstats import gmean
from psychacoustics import Chroma as ChromaScale, Bark as BarkScale
from zounds.nputil import safe_log
from zounds.timeseries import ConstantRateTimeSeries, SR44100


class FFT(Node):
    def __init__(self, needs=None, axis=-1):
        super(FFT, self).__init__(needs=needs)
        self._axis = axis

    def _process(self, data):
        transformed = np.fft.fft(data, axis=self._axis)
        sl = [slice(None) for _ in xrange(len(transformed.shape))]
        positive = data.shape[self._axis] // 2
        sl[self._axis] = slice(0, positive, None)
        yield ConstantRateTimeSeries(
                transformed[sl],
                data.frequency,
                data.duration)


class DCT(Node):
    def __init__(self, needs=None, axis=-1):
        super(DCT, self).__init__(needs=needs)
        self._axis = axis

    def _process(self, data):
        yield ConstantRateTimeSeries( \
                dct(data, norm='ortho', axis=self._axis),
                data.frequency,
                data.duration)


# TODO: This constructor should not take a samplerate; that information should
# be encapsulated in the data that's passed in
class Chroma(Node):
    def __init__( \
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

        yield ConstantRateTimeSeries( \
                self._chroma_scale.transform(data),
                data.frequency,
                data.duration)


# TODO: This constructor should not take a samplerate; that information should
# be encapsulated in the data that's passed in
class BarkBands(Node):
    def __init__( \
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
            self._bark_scale = BarkScale( \
                    self._samplerate.samples_per_second,
                    data.shape[1] * 2,
                    self._n_bands,
                    self._start_freq_hz,
                    self._stop_freq_hz)
        yield ConstantRateTimeSeries( \
                self._bark_scale.transform(data),
                data.frequency,
                data.duration)


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
        yield ConstantRateTimeSeries(
            (data * self._bins).sum(axis=1) / self._bins_sum,
            data.frequency,
            data.duration)


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
        m = data.mean(axis=1)
        m[m == 0] = -1e5
        yield ConstantRateTimeSeries(
            gmean(data, axis=1) / m,
            data.frequency,
            data.duration)


class BFCC(Node):
    def __init__(self, needs=None, n_coeffs=13, exclude=1):
        super(BFCC, self).__init__(needs=needs)
        self._n_coeffs = n_coeffs
        self._exclude = exclude

    def _process(self, data):
        data = np.abs(data)
        bfcc = dct(safe_log(data), axis=1) \
            [:, self._exclude: self._exclude + self._n_coeffs]
        yield ConstantRateTimeSeries( \
                bfcc.copy(), data.frequency, data.duration)
