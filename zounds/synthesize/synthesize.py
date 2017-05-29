from __future__ import division
import numpy as np
from scipy.fftpack import idct
from zounds.timeseries import audio_sample_rate, Seconds, AudioSamples
from zounds.spectral import DCTIV
from zounds.spectral.sliding_window import \
    IdentityWindowingFunc, OggVorbisWindowingFunc
from zounds.core import ArrayWithUnits, IdentityDimension


class ShortTimeTransformSynthesizer(object):
    def __init__(self):
        super(ShortTimeTransformSynthesizer, self).__init__()

    def _transform(self, frames):
        return frames

    def _windowing_function(self):
        return IdentityWindowingFunc()

    def _overlap_add(self, frames):
        time_dim = frames.dimensions[0]
        sample_length_seconds = time_dim.duration_in_seconds / frames.shape[-1]
        samples_per_second = int(1 / sample_length_seconds)
        samplerate = audio_sample_rate(samples_per_second)
        windowsize = int(np.round(time_dim.duration / samplerate.frequency))
        hopsize = int(np.round(time_dim.frequency / samplerate.frequency))
        arr = np.zeros(int(time_dim.end / samplerate.frequency))
        for i, f in enumerate(frames):
            start = i * hopsize
            stop = start + windowsize
            l = len(arr[start:stop])
            arr[start:stop] += (self._windowing_function() * f[:l])
        return AudioSamples(arr, samplerate)

    def synthesize(self, frames):
        audio = self._transform(frames)
        ts = ArrayWithUnits(audio, [frames.dimensions[0], IdentityDimension()])
        return self._overlap_add(ts)


class FFTSynthesizer(ShortTimeTransformSynthesizer):
    def __init__(self):
        super(FFTSynthesizer, self).__init__()

    def _windowing_function(self):
        return OggVorbisWindowingFunc()

    def _transform(self, frames):
        return np.fft.fftpack.irfft(frames, norm='ortho')


class DCTSynthesizer(ShortTimeTransformSynthesizer):
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
        tick = np.random.random_sample(int(sr * .1))
        tick *= np.linspace(1, 0, len(tick))
        # create silence
        samples = np.zeros(int(sr * (duration / Seconds(1))))
        ticks_per_second = Seconds(1) / tick_frequency
        # introduce periodic ticking sound
        step = int(sr // ticks_per_second)
        print 'STEP', step
        for i in xrange(0, len(samples), step):
            print 'STEP 2', i
            samples[i:i + len(tick)] = tick
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
        return AudioSamples(
                np.random.random_sample(int(sr * seconds)), self.samplerate)
