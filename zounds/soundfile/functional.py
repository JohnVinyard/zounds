from zounds.timeseries import AudioSamples
from resample import Resampler
import numpy as np


def resample(samples, new_sample_rate):
    if new_sample_rate == samples.samplerate:
        return samples
    rs = Resampler(new_sample_rate)
    new_samples = np.concatenate(list(rs._process(samples)))
    return AudioSamples(new_samples, new_sample_rate)
