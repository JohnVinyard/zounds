from __future__ import division
import numpy as np
from scipy.fftpack import idct
from zounds.timeseries import \
    audio_sample_rate, AudioSamples, ConstantRateTimeSeries, Seconds


class ShortTimeTransformSynthesizer(object):
    def __init__(self):
        super(ShortTimeTransformSynthesizer, self).__init__()

    def _transform(self, frames):
        return frames

    def _overlap_add(self, frames):
        sample_length_seconds = frames.duration_in_seconds / frames.shape[-1]
        samples_per_second = int(1 / sample_length_seconds)
        samplerate = audio_sample_rate(samples_per_second)
        windowsize = int(frames.duration / samplerate.frequency)
        hopsize = int(frames.frequency / samplerate.frequency)
        arr = np.zeros(frames.end / samplerate.frequency)
        for i, f in enumerate(frames):
            start = i * hopsize
            stop = start + windowsize
            arr[start:stop] += f
        return AudioSamples(arr, samplerate)

    def synthesize(self, frames):
        audio = self._transform(frames)
        ts = ConstantRateTimeSeries.from_example(audio, frames)
        return self._overlap_add(ts)


class FFTSynthesizer(ShortTimeTransformSynthesizer):
    def __init__(self):
        super(FFTSynthesizer, self).__init__()

    def _transform(self, frames):
        return np.fft.fftpack.ifft(frames)


class DCTSynthesizer(ShortTimeTransformSynthesizer):
    def __init__(self):
        super(DCTSynthesizer, self).__init__()

    def _transform(self, frames):
        return idct(frames, norm='ortho')


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
        samples = np.zeros(sr * (duration / Seconds(1)))
        ticks_per_second = Seconds(1) / tick_frequency
        # introduce periodic ticking sound
        step = int(sr // ticks_per_second)
        for i in xrange(0, len(samples), step):
            samples[i:i+len(tick)] = tick
        return AudioSamples(samples, self.samplerate)
