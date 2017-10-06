from __future__ import division

import numpy as np
from scipy.fftpack import dct, idct

from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.spectral import DCTIV, LinearScale
from zounds.spectral import FrequencyDimension
from zounds.spectral.sliding_window import \
    IdentityWindowingFunc, OggVorbisWindowingFunc
from zounds.timeseries import \
    nearest_audio_sample_rate, Seconds, AudioSamples


class ShortTimeTransformSynthesizer(object):
    def __init__(self):
        super(ShortTimeTransformSynthesizer, self).__init__()

    def _transform(self, frames):
        return frames

    def _windowing_function(self):
        return IdentityWindowingFunc()

    def _overlap_add(self, frames):
        time_dim = frames.dimensions[0]
        sample_freq = time_dim.duration / frames.shape[-1]
        windowsize = int(np.round(time_dim.duration / sample_freq))
        hopsize = int(np.round(time_dim.frequency / sample_freq))

        # create an empty array of audio samples
        arr = np.zeros(int(time_dim.end / sample_freq))
        for i, f in enumerate(frames):
            start = i * hopsize
            stop = start + windowsize
            l = len(arr[start:stop])
            arr[start:stop] += (self._windowing_function() * f[:l])

        sr = nearest_audio_sample_rate(Seconds(1) / sample_freq)
        return AudioSamples(arr, sr)

    def synthesize(self, frames):
        audio = self._transform(frames)
        ts = ArrayWithUnits(audio, [frames.dimensions[0], IdentityDimension()])
        return self._overlap_add(ts)


class WindowedAudioSynthesizer(ShortTimeTransformSynthesizer):
    def __init__(self):
        super(WindowedAudioSynthesizer, self).__init__()


class FFTSynthesizer(ShortTimeTransformSynthesizer):
    """
    Inverts the short-time fourier transform, e.g. the output of the
    :class:`zounds.spectral.FFT` processing node.

    Here's an example that extracts a short-time fourier transform, and then
    inverts it.

    .. code:: python

        import zounds

        STFT = zounds.stft(
            resample_to=zounds.SR11025(),
            store_fft=True)


        @zounds.simple_in_memory_settings
        class Sound(STFT):
            pass

        # produce some additive sine waves
        sine_synth = zounds.SineSynthesizer(zounds.SR22050())
        samples = sine_synth.synthesize(
            zounds.Seconds(4), freqs_in_hz=[220, 400, 880])

        # process the sound, including a short-time fourier transform feature
        _id = Sound.process(meta=samples.encode())
        snd = Sound(_id)

        # invert the frequency-domain feature to reover the original audio
        fft_synth = zounds.FFTSynthesizer()
        recon = fft_synth.synthesize(snd.fft)
        print recon.__class__  #  AudioSamples instance with reconstructed audio

    See Also:
        :class:`zounds.spectral.FFT`
    """
    def __init__(self):
        super(FFTSynthesizer, self).__init__()

    def _windowing_function(self):
        return OggVorbisWindowingFunc()

    def _transform(self, frames):
        return np.fft.fftpack.irfft(frames, norm='ortho')


class DCTSynthesizer(ShortTimeTransformSynthesizer):
    """
    Inverts the short-time discrete cosine transform (type II), e.g., the output
    of the :class:`zounds.spectral.DCT` processing node

    Here's an example that extracts a short-time discrete cosine transform, and
    then inverts it.

    .. code:: python

        import zounds

        Resampled = zounds.resampled(resample_to=zounds.SR11025())


        @zounds.simple_in_memory_settings
        class Sound(Resampled):
            windowed = zounds.ArrayWithUnitsFeature(
                zounds.SlidingWindow,
                needs=Resampled.resampled,
                wscheme=zounds.HalfLapped(),
                wfunc=zounds.OggVorbisWindowingFunc(),
                store=False)

            dct = zounds.ArrayWithUnitsFeature(
                zounds.DCT,
                needs=windowed,
                store=True)

        # produce some additive sine waves
        sine_synth = zounds.SineSynthesizer(zounds.SR22050())
        samples = sine_synth.synthesize(
            zounds.Seconds(4), freqs_in_hz=[220, 400, 880])

        # process the sound, including a short-time fourier transform feature
        _id = Sound.process(meta=samples.encode())
        snd = Sound(_id)

        # invert the frequency-domain feature to reover the original audio
        dct_synth = zounds.DCTSynthesizer()
        recon = dct_synth.synthesize(snd.dct)
        print recon.__class__  # AudioSamples instance with reconstructed audio

    See Also:
        :class:`zounds.spectral.DCT`
    """
    def __init__(self, windowing_func=IdentityWindowingFunc()):
        super(DCTSynthesizer, self).__init__()
        self.windowing_func = windowing_func

    def _windowing_function(self):
        return self.windowing_func

    def _transform(self, frames):
        return idct(frames, norm='ortho')


class DCTIVSynthesizer(ShortTimeTransformSynthesizer):
    """
    Perform the inverse of the DCTIV transform, which is the same as the forward
    transformation
    """

    def __init__(self, windowing_func=IdentityWindowingFunc()):
        super(DCTIVSynthesizer, self).__init__()
        self.windowing_func = windowing_func

    def _windowing_function(self):
        return self.windowing_func

    def _transform(self, frames):
        return list(DCTIV()._process(frames))[0]


class MDCTSynthesizer(ShortTimeTransformSynthesizer):
    def __init__(self):
        super(MDCTSynthesizer, self).__init__()

    def _windowing_function(self):
        return OggVorbisWindowingFunc()

    def _transform(self, frames):
        l = frames.shape[1]
        t = np.arange(0, 2 * l)
        f = np.arange(0, l)
        cpi = -1j * np.pi
        a = frames * np.exp(cpi * (f + 0.5) * (l + 1) / 2 / l)
        b = np.fft.fft(a, 2 * l)
        return np.sqrt(2 / l) * np.real(b * np.exp(cpi * t / 2 / l))


class BaseFrequencyAdaptiveSynthesizer(object):
    def __init__(
            self,
            scale,
            band_transform,
            short_time_synth,
            samplerate,
            coeffs_dtype,
            scale_slices_always_even):
        super(BaseFrequencyAdaptiveSynthesizer, self).__init__()
        self.scale_slices_always_even = scale_slices_always_even
        self.coeffs_dtype = coeffs_dtype
        self.scale = scale
        self.samplerate = samplerate
        self.short_time_synth = short_time_synth
        self.band_transform = band_transform

    def _n_linear_scale_bands(self, frequency_adaptive_coeffs):
        raise NotImplementedError()

    def synthesize(self, freq_adaptive_coeffs):
        fac = freq_adaptive_coeffs

        linear_scale = LinearScale.from_sample_rate(
            self.samplerate,
            self._n_linear_scale_bands(fac),
            always_even=self.scale_slices_always_even)

        frequency_dimension = FrequencyDimension(linear_scale)

        coeffs = ArrayWithUnits(
            np.zeros((len(fac), linear_scale.n_bands), dtype=self.coeffs_dtype),
            dimensions=[fac.dimensions[0], frequency_dimension])

        for band in self.scale:
            coeffs[:, band] += self.band_transform(fac[:, band], norm='ortho')

        return self.short_time_synth.synthesize(coeffs)


class FrequencyAdaptiveDCTSynthesizer(BaseFrequencyAdaptiveSynthesizer):
    def __init__(self, scale, samplerate):
        super(FrequencyAdaptiveDCTSynthesizer, self).__init__(
            scale,
            dct,
            DCTSynthesizer(),
            samplerate,
            np.float64,
            scale_slices_always_even=True)

    def _n_linear_scale_bands(self, frequency_adaptive_coeffs):
        fac = frequency_adaptive_coeffs.dimensions[0]
        return int(fac.duration / self.samplerate.frequency)


class FrequencyAdaptiveFFTSynthesizer(BaseFrequencyAdaptiveSynthesizer):
    def __init__(self, scale, samplerate):
        super(FrequencyAdaptiveFFTSynthesizer, self).__init__(
            scale,
            np.fft.rfft,
            FFTSynthesizer(),
            samplerate,
            np.complex128,
            scale_slices_always_even=False)

    def _n_linear_scale_bands(self, frequency_adaptive_coeffs):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft
        fac = frequency_adaptive_coeffs.dimensions[0]
        raw_samples = int(fac.duration / self.samplerate.frequency)
        return int(raw_samples // 2) + 1


class SineSynthesizer(object):
    """
    Synthesize sine waves
    """

    def __init__(self, samplerate):
        super(SineSynthesizer, self).__init__()
        self.samplerate = samplerate

    def synthesize(self, duration, freqs_in_hz=[440.]):
        freqs = np.array(freqs_in_hz)
        scaling = 1 / len(freqs)
        sr = int(self.samplerate)
        cps = freqs / sr
        ts = (duration / Seconds(1)) * sr
        ranges = np.array([np.arange(0, ts * c, c) for c in cps])
        raw = (np.sin(ranges * (2 * np.pi)) * scaling).sum(axis=0)
        return AudioSamples(raw, self.samplerate)


class TickSynthesizer(object):
    """
    Synthesize short, percussive, periodic "ticks"
    """

    def __init__(self, samplerate):
        super(TickSynthesizer, self).__init__()
        self.samplerate = samplerate

    def synthesize(self, duration, tick_frequency):
        sr = self.samplerate.samples_per_second
        # create a short, tick sound
        tick = np.random.uniform(low=-1., high=1., size=int(sr * .1))
        tick *= np.linspace(1, 0, len(tick))
        # create silence
        samples = np.zeros(int(sr * (duration / Seconds(1))))
        ticks_per_second = Seconds(1) / tick_frequency
        # introduce periodic ticking sound
        step = int(sr // ticks_per_second)
        for i in xrange(0, len(samples), step):
            size = len(samples[i:i + len(tick)])
            samples[i:i + len(tick)] += tick[:size]
        return AudioSamples(samples, self.samplerate)


class NoiseSynthesizer(object):
    """
    Synthesize white noise
    """

    def __init__(self, samplerate):
        super(NoiseSynthesizer, self).__init__()
        self.samplerate = samplerate

    def synthesize(self, duration):
        sr = self.samplerate.samples_per_second
        seconds = duration / Seconds(1)
        samples = np.random.uniform(low=-1., high=1., size=int(sr * seconds))
        return AudioSamples(samples, self.samplerate)


class SilenceSynthesizer(object):
    def __init__(self, samplerate):
        super(SilenceSynthesizer, self).__init__()
        self.samplerate = samplerate

    def synthesize(self, duration):
        sr = self.samplerate.samples_per_second
        seconds = duration / Seconds(1)
        samples = np.zeros(int(sr * seconds))
        return AudioSamples(samples, self.samplerate)
