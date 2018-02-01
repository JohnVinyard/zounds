from __future__ import division
from frequencyscale import LinearScale, FrequencyBand, ExplicitScale
from tfrepresentation import FrequencyDimension
from frequencyadaptive import FrequencyAdaptive
from zounds.timeseries import audio_sample_rate, TimeSlice, Seconds
from zounds.core import ArrayWithUnits, IdentityDimension
from sliding_window import IdentityWindowingFunc
from zounds.loudness import log_modulus, unit_scale
import numpy as np
from scipy.signal import resample


def fft(x, axis=-1, padding_samples=0):
    """
    Apply an FFT along the given dimension, and with the specified amount of
    zero-padding

    Args:
        x (ArrayWithUnits): an :class:`~zounds.core.ArrayWithUnits` instance
            which has one or more :class:`~zounds.timeseries.TimeDimension`
            axes
        axis (int): The axis along which the fft should be applied
        padding_samples (int): The number of padding zeros to apply along
            axis before performing the FFT
    """
    if padding_samples > 0:
        padded = np.concatenate(
            [x, np.zeros((len(x), padding_samples), dtype=x.dtype)],
            axis=axis)
    else:
        padded = x

    transformed = np.fft.rfft(padded, axis=axis, norm='ortho')
    sr = audio_sample_rate(int(Seconds(1) / x.dimensions[axis].frequency))
    scale = LinearScale.from_sample_rate(sr, transformed.shape[-1])
    new_dimensions = list(x.dimensions)
    new_dimensions[axis] = FrequencyDimension(scale)
    return ArrayWithUnits(transformed, new_dimensions)


def stft(x, window_sample_rate, window=None):
    duration = TimeSlice(window_sample_rate.duration)
    frequency = TimeSlice(window_sample_rate.frequency)
    _, arr = x.sliding_window_with_leftovers(
        duration, frequency, dopad=True)
    window = window or IdentityWindowingFunc()
    windowed = arr * window
    return fft(windowed)


def phase_shift(coeffs, samplerate, time_shift, axis=-1, frequency_band=None):
    frequency_dim = coeffs.dimensions[axis]
    if not isinstance(frequency_dim, FrequencyDimension):
        raise ValueError(
            'dimension {axis} of coeffs must be a FrequencyDimension instance, '
            'but was {cls}'.format(axis=axis, cls=frequency_dim.__class__))

    n_coeffs = coeffs.shape[axis]
    shift_samples = int(time_shift / samplerate.frequency)
    shift = (np.arange(0, n_coeffs) * 2j * np.pi) / n_coeffs
    shift = np.exp(-shift * shift_samples)
    shift = ArrayWithUnits(shift, [frequency_dim])

    frequency_band = frequency_band or slice(None)
    new_coeffs = coeffs.copy()

    if coeffs.ndim == 1:
        new_coeffs[frequency_band] *= shift[frequency_band]
        return new_coeffs

    slices = [slice(None) for _ in xrange(coeffs.ndim)]
    slices[axis] = frequency_band
    new_coeffs[tuple(slices)] *= shift[frequency_band]
    return new_coeffs


# def apply_scale(short_time_fft, scale, reducer=np.sum, window=None):
#     magnitudes = np.abs(short_time_fft.real)
#     output = np.zeros(
#         short_time_fft.shape[:-1] + (len(scale),), dtype=magnitudes.dtype)
#     output = ArrayWithUnits(
#         output, short_time_fft.dimensions[:-1] + (FrequencyDimension(scale),))
#     window = window or IdentityWindowingFunc()
#     for i, freq_band in enumerate(scale):
#         reduced_band = reducer(magnitudes[..., freq_band] * window, axis=-1)
#         output[..., i] = reduced_band
#     return output


def apply_scale(short_time_fft, scale, window=None):
    magnitudes = np.abs(short_time_fft.real)
    spectrogram = scale.apply(magnitudes, window)
    dimensions = short_time_fft.dimensions[:-1] + (FrequencyDimension(scale),)
    return ArrayWithUnits(spectrogram, dimensions)


def rainbowgram(time_frequency_repr, colormap):

    # magnitudes on a log scale, and shifted and
    # scaled to the unit interval
    magnitudes = np.abs(time_frequency_repr.real)
    magnitudes = log_modulus(magnitudes * 1000)
    magnitudes = unit_scale(magnitudes)

    angles = np.angle(time_frequency_repr)
    angles = np.unwrap(angles, axis=0)
    angles = np.gradient(angles)[0]
    angles = unit_scale(angles)

    colors = colormap(angles)
    colors *= magnitudes[..., None]

    # exclude the alpha channel, if there is one
    colors = colors[..., :3]
    arr = ArrayWithUnits(
        colors, time_frequency_repr.dimensions + (IdentityDimension(),))
    return arr


def frequency_decomposition(x, sizes):
    sizes = sorted(sizes)

    original_size = x.shape[-1]
    time_dimension = x.dimensions[-1]
    samplerate = audio_sample_rate(time_dimension.samples_per_second)
    data = x.copy()

    bands = []
    frequency_bands = []
    start_hz = 0

    for size in sizes:
        if size != original_size:
            s = resample(data, size, axis=-1)
        else:
            s = data

        bands.append(s)
        data -= resample(s, original_size, axis=-1)

        stop_hz = samplerate.nyquist * (size / original_size)
        frequency_bands.append(FrequencyBand(start_hz, stop_hz))
        start_hz = stop_hz

    scale = ExplicitScale(frequency_bands)
    return FrequencyAdaptive(bands, scale=scale, time_dimension=x.dimensions[0])
