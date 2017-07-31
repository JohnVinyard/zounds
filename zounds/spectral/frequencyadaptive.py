import numpy as np
from tfrepresentation import ExplicitFrequencyDimension
from zounds.core import ArrayWithUnits
from scipy.signal import resample


class FrequencyAdaptive(ArrayWithUnits):
    def __new__(cls, arrs, time_dimension, scale):
        stops = list(np.cumsum([arr.shape[1] for arr in arrs]))
        slices = [slice(start, stop)
                  for (start, stop) in zip([0] + stops, stops)]
        dimensions = [time_dimension, ExplicitFrequencyDimension(scale, slices)]
        array = np.concatenate(arrs, axis=1)
        return ArrayWithUnits.__new__(cls, array, dimensions)

    @property
    def scale(self):
        return self.dimensions[1].scale

    def square(self, n_coeffs):
        return np.vstack(
            [resample(band, n_coeffs) for band in self.iter_bands()]).T

    def iter_bands(self):
        return (self[:, band] for band in self.scale)

    @classmethod
    def from_array_with_units(cls, arr):
        fdim = arr.dimensions[1]
        arrs = [arr[:, band] for band in fdim.scale]
        fa = FrequencyAdaptive(arrs, arr.dimensions[0], fdim.scale)
        return fa
