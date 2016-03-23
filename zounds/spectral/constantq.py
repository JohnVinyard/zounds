from __future__ import division
import numpy as np
from zounds.timeseries import ConstantRateTimeSeries, SR44100
from zounds.visualize import plot


class OctaveScale(object):
    def __init__(self, freq_min, freq_max, bands_per_octave):
        self.bands = int( \
                np.ceil(np.log2(freq_max / freq_min) * bands_per_octave)) + 1
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.bands_per_octave = bands_per_octave
        self.pow2n = 2 ** (1 / self.bands_per_octave)
        self.quality_factor = np.sqrt(self.pow2n) / (self.pow2n - 1) / 2
        self.center_frequencies = \
            self.freq_min * self.pow2n ** np.arange(self.bands)


class Windows(object):
    def __init__(self, scale, timeseries):
        self.scale = scale
        self.samplerate = timeseries.samples_per_second
        print self.samplerate
        self.nsamples = len(timeseries)
        self.q = scale.quality_factor
        nyquist = self.samplerate / 2
        print nyquist
        cf = self.scale.center_frequencies
        # limit frequencies to those greater than zero, and less than the
        # nyquist frequency
        cf = cf[(cf > 0) & (cf <= nyquist)]

        q_needed = cf * (len(timeseries) / (8 * self.samplerate))
        if np.any(self.q > q_needed):
            raise Exception('Q factor too high')

        freqs = np.concatenate([(0,), cf, (nyquist,)])
        self.fbas = np.concatenate([freqs, self.samplerate - freqs[-2:0:-1]])
        self.fbas2 = self.fbas * (len(timeseries) / self.samplerate)


if __name__ == '__main__':
    scale = OctaveScale(20, 2e4, 12)
    sr = SR44100()
    ts = ConstantRateTimeSeries( \
            np.random.random_sample(44100 * 8), sr.frequency, sr.duration)
    windows = Windows(scale, ts)
    plot(windows.fbas, '/home/john/Desktop/fbas.png')
    plot(windows.fbas2, '/home/john/Desktop/fbas2.png')
