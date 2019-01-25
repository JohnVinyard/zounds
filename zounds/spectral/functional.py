from __future__ import division
from frequencyscale import LinearScale, FrequencyBand, ExplicitScale
from tfrepresentation import FrequencyDimension
from frequencyadaptive import FrequencyAdaptive
from zounds.timeseries import \
    audio_sample_rate, TimeSlice, Seconds, TimeDimension, HalfLapped, \
    Milliseconds, SampleRate
from zounds.core import ArrayWithUnits, IdentityDimension
from sliding_window import \
    IdentityWindowingFunc, HanningWindowingFunc, WindowingFunc
from zounds.loudness import log_modulus, unit_scale
import numpy as np
from scipy.signal import resample, firwin2
from matplotlib import cm
from scipy.signal import hann, morlet
from itertools import repeat
from zounds.nputil import sliding_window


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


def stft(x, window_sample_rate=HalfLapped(), window=HanningWindowingFunc()):
    duration = TimeSlice(window_sample_rate.duration)
    frequency = TimeSlice(window_sample_rate.frequency)

    if x.ndim == 1:
        _, arr = x.sliding_window_with_leftovers(
            duration, frequency, dopad=True)
    elif x.ndim == 2 and isinstance(x.dimensions[0], IdentityDimension):
        arr = x.sliding_window((1, duration), (1, frequency))
        td = x.dimensions[-1]
        dims = [IdentityDimension(), TimeDimension(*window_sample_rate), td]
        arr = ArrayWithUnits(arr.reshape((len(x), -1, arr.shape[-1])), dims)
    else:
        raise ValueError(
            'x must either have a single TimeDimension, or '
            '(IdentityDimension, TimeDimension)')

    window = window or IdentityWindowingFunc()
    windowed = arr * window._wdata(arr.shape[-1])
    return fft(windowed)


def time_stretch(x, factor, frame_sample_rate=None):
    if frame_sample_rate is None:
        sr = HalfLapped()
        sr = SampleRate(frequency=sr.frequency / 2, duration=sr.duration)
    else:
        sr = frame_sample_rate

    hop_length, window_length = sr.discrete_samples(x)

    win = WindowingFunc(windowing_func=hann)

    # to simplify, let's always compute the stft in "batch" mode
    if x.ndim == 1:
        x = x.reshape((1,) + x.shape)

    D = stft(x, sr, win)

    n_fft_coeffs = D.shape[-1]
    n_frames = D.shape[1]
    n_batches = D.shape[0]

    time_steps = np.arange(0, n_frames, factor, dtype=np.float)

    weights = np.mod(time_steps, 1.0)

    exp_phase_advance = np.linspace(0, np.pi * hop_length, n_fft_coeffs)

    # pad in the time dimension, so no edge/end frames are left out
    # coeffs = np.pad(D, [(0, 0), (0, 2), (0, 0)], mode='constant')
    shape = list(D.shape)
    shape[1] += 2
    coeffs = np.zeros(shape, dtype=D.dtype)
    coeffs[:, :-2, :] = D

    coeffs_mags = np.abs(coeffs)
    coeffs_phases = np.angle(coeffs)

    # we need a phase accumulator for every item in the batch
    phase_accum = coeffs_phases[:, :1, :]

    sliding_indices = np.vstack([time_steps, time_steps + 1]).T.astype(np.int32)

    windowed_mags = coeffs_mags[:, sliding_indices, :]
    windowed_phases = coeffs_phases[:, sliding_indices, :]

    first_mags = windowed_mags[:, :, 0, :]
    second_mags = windowed_mags[:, :, 1, :]

    first_phases = windowed_phases[:, :, 0, :]
    second_phases = windowed_phases[:, :, 1, :]

    # compute all the phase stuff
    two_pi = 2.0 * np.pi
    dphase = (second_phases - first_phases - exp_phase_advance)
    dphase -= two_pi * np.round(dphase / two_pi)
    dphase += exp_phase_advance

    all_phases = np.concatenate([phase_accum, dphase], axis=1)
    dphase = np.cumsum(all_phases, axis=1, out=all_phases)
    dphase = dphase[:, :-1, :]

    # linear interpolation of FFT coefficient magnitudes
    weights = weights[None, :, None]
    mags = ((1.0 - weights) * first_mags) + (weights * second_mags)

    # combine magnitudes and phases
    new_coeffs = mags * np.exp(1.j * dphase)

    # synthesize the new frames
    new_frames = np.fft.irfft(new_coeffs, axis=-1, norm='ortho')
    # new_frames = new_frames * win._wdata(new_frames.shape[-1])
    new_frames = np.multiply(
        new_frames, win._wdata(new_frames.shape[-1]), out=new_frames)

    # overlap add the new audio samples
    new_n_samples = int(x.shape[-1] / factor)
    output = np.zeros((n_batches, new_n_samples), dtype=x.dtype)
    for i in xrange(new_frames.shape[1]):
        start = i * hop_length
        stop = start + new_frames.shape[-1]
        l = output[:, start: stop].shape[1]
        output[:, start: stop] += new_frames[:, i, :l]

    return ArrayWithUnits(output, [IdentityDimension(), x.dimensions[-1]])


def pitch_shift(x, semitones, frame_sample_rate=None):
    original_shape = x.shape[1] if x.ndim == 2 else x.shape[0]

    # first, perform a time stretch so that the audio will have the desired
    # pitch
    factor = 2.0 ** (-float(semitones) / 12.0)
    stretched = time_stretch(x, factor, frame_sample_rate=frame_sample_rate)

    # hang on to original dimensions
    dimensions = stretched.dimensions

    # window the audio using a power-of-2 frame size for more efficient FFT
    # computations
    batch_size = stretched.shape[0]
    window_size = 1024
    step = (1, window_size)
    new_window_shape = int(window_size * factor)
    padding = window_size - int(stretched.shape[-1] % window_size)
    stretched = np.pad(stretched, ((0, 0), (0, padding)), mode='constant')
    windowed = sliding_window(stretched, step, step, flatten=False).squeeze()

    # resample the audio so that it has the correct duration
    rs = resample(windowed, new_window_shape, axis=-1)

    # flatten out the windowed, resampled audio
    rs = rs.reshape(batch_size, -1)

    # slice the audio to remove residual zeros resulting from our power-of-2
    # zero padding above
    rs = rs[:, :original_shape]

    return ArrayWithUnits(rs, dimensions)


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


def apply_scale(short_time_fft, scale, window=None):
    magnitudes = np.abs(short_time_fft.real)
    spectrogram = scale.apply(magnitudes, window)
    dimensions = short_time_fft.dimensions[:-1] + (FrequencyDimension(scale),)
    return ArrayWithUnits(spectrogram, dimensions)


def rainbowgram(time_frequency_repr, colormap=cm.rainbow):
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


def fir_filter_bank(scale, taps, samplerate, window):
    basis = np.zeros((len(scale), taps))
    basis = ArrayWithUnits(basis, [
        FrequencyDimension(scale),
        TimeDimension(*samplerate)])

    nyq = samplerate.nyquist

    if window.ndim == 1:
        window = repeat(window, len(scale))

    for i, band, win in zip(xrange(len(scale)), scale, window):
        start_hz = max(0, band.start_hz)
        stop_hz = min(nyq, band.stop_hz)
        freqs = np.linspace(
            start_hz / nyq, stop_hz / nyq, len(win))
        freqs = [0] + list(freqs) + [1]
        gains = [0] + list(win) + [0]
        basis[i] = firwin2(taps, freqs, gains)

    return basis


def morlet_filter_bank(
        samplerate,
        kernel_size,
        scale,
        scaling_factor,
        normalize=True):
    """
    Create a bank of finite impulse response filters, with
    frequencies centered on the sub-bands of scale
    """
    basis_size = len(scale)
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    try:
        if len(scaling_factor) != len(scale):
            raise ValueError('scaling factor must have same length as scale')
    except TypeError:
        scaling_factor = np.repeat(float(scaling_factor), len(scale))

    sr = int(samplerate)

    for i, band in enumerate(scale):
        scaling = scaling_factor[i]
        w = band.center_frequency / (scaling * 2 * sr / kernel_size)
        basis[i] = morlet(
            M=kernel_size,
            w=w,
            s=scaling)
    basis = basis.real

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8

    basis = ArrayWithUnits(
        basis, [FrequencyDimension(scale), TimeDimension(*samplerate)])

    return basis


def auto_correlogram(x, filter_bank, correlation_window=Milliseconds(30)):
    n_filters = filter_bank.shape[0]
    filter_size = filter_bank.shape[1]

    corr_win_samples = int(correlation_window / x.samplerate.frequency)
    windowed = sliding_window(x, filter_size, 1, flatten=False)
    print windowed.shape
    filtered = np.dot(windowed, filter_bank.T)
    print filtered.shape
    corr = sliding_window(
        filtered,
        ws=(corr_win_samples, n_filters),
        ss=(1, n_filters),
        flatten=False)
    print corr.shape

    padded_shape = list(corr.shape)
    padded_shape[2] = corr_win_samples * 2
    padded = np.zeros(padded_shape, dtype=np.float32)
    padded[:, :, :corr_win_samples, :] = corr
    print padded.shape

    coeffs = np.fft.fft(padded, axis=2, norm='ortho')
    correlated = np.fft.ifft(np.abs(coeffs) ** 2, axis=2, norm='ortho')
    return np.concatenate([
        correlated[:, :, corr_win_samples:, :],
        correlated[:, :, :corr_win_samples, :],
    ], axis=2)
    return correlated


def dct_basis(size):
    r = np.arange(size)
    basis = np.outer(r, r + 0.5)
    basis = np.cos((np.pi / size) * basis)
    return basis


def frequency_decomposition(x, sizes):
    sizes = sorted(sizes)

    if x.ndim == 1:
        end = x.dimensions[0].end
        x = ArrayWithUnits(
            x[None, ...], [TimeDimension(end, end), x.dimensions[0]])

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
            s = data.copy()

        bands.append(s)
        data -= resample(s, original_size, axis=-1)

        stop_hz = samplerate.nyquist * (size / original_size)
        frequency_bands.append(FrequencyBand(start_hz, stop_hz))
        start_hz = stop_hz

    scale = ExplicitScale(frequency_bands)
    return FrequencyAdaptive(bands, scale=scale, time_dimension=x.dimensions[0])
