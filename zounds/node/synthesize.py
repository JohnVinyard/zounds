from __future__ import division
import numpy as np
from scipy.fftpack import idct
from samplerate import audio_sample_rate
from audiosamples import AudioSamples


class ShortTimeTransformSynthesizer(object):

    def __init__(self):
        super(ShortTimeTransformSynthesizer, self).__init__()

    def _transform(self, frames):
        return frames

    def _overlap_add(self, frames):
        samples_per_second = frames.duration_in_seconds / frames.shape[-1]
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
        return self._overlap_add(audio)


class ShortTimeFourierTransformSynthesizer(ShortTimeTransformSynthesizer):

    def __init__(self):
        super(ShortTimeFourierTransformSynthesizer, self).__init__()

    def _transform(self, frames):
        return np.fft.fftpack.ifft(frames)


class ShortTimeDiscreteCosineTransformSynthesizer(ShortTimeTransformSynthesizer):

    def __init__(self):
        super(ShortTimeDiscreteCosineTransformSynthesizer, self).__init__()

    def _transform(self, frames):
        return idct(frames, norm='ortho')
