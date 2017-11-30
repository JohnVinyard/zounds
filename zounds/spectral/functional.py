from frequencyscale import LinearScale
from tfrepresentation import FrequencyDimension
from zounds.timeseries import audio_sample_rate, TimeSlice
from zounds.core import ArrayWithUnits
from sliding_window import IdentityWindowingFunc
import numpy as np


def fft(x, axis=-1):
    transformed = np.fft.rfft(x, axis=axis, norm='ortho')

    sr = audio_sample_rate(
        int(x.shape[1] / x.dimensions[0].duration_in_seconds))
    scale = LinearScale.from_sample_rate(sr, transformed.shape[-1])

    return ArrayWithUnits(
        transformed, [x.dimensions[0], FrequencyDimension(scale)])


def stft(x, window_sample_rate, window=None):
    duration = TimeSlice(window_sample_rate.duration)
    frequency = TimeSlice(window_sample_rate.frequency)
    _, arr = x.sliding_window_with_leftovers(
        duration, frequency, dopad=True)
    window = window or IdentityWindowingFunc()
    windowed = arr * window
    return fft(windowed)


def apply_scale(short_time_fft, scale, reducer=np.sum, window=None):
    magnitudes = np.abs(short_time_fft.real)
    output = np.zeros(
        short_time_fft.shape[:-1] + (len(scale),), dtype=magnitudes.dtype)
    output = ArrayWithUnits(
        output, short_time_fft.dimensions[:-1] + (FrequencyDimension(scale),))
    window = window or IdentityWindowingFunc()
    for i, freq_band in enumerate(scale):
        reduced_band = reducer(magnitudes[..., freq_band] * window, axis=-1)
        output[..., i] = reduced_band
    return output
