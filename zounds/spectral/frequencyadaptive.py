from __future__ import division

import numpy as np
from scipy.signal import resample

from tfrepresentation import ExplicitFrequencyDimension, FrequencyDimension
from zounds.core import ArrayWithUnits
from zounds.timeseries import ConstantRateTimeSeries
from zounds.timeseries import Picoseconds, TimeDimension


class FrequencyAdaptive(ArrayWithUnits):
    """
    TODO: This needs some love. Mutually exclusive constructor arguments are no
    bueno

    Args:
        arrs: TODO
        time_dimension (TimeDimension): the time dimension of the first axis of
            this array
        scale (FrequencyScale): The frequency scale corresponding to the first
            axis of this array, mutually exclusive with the
            :code:`explicit_freq_dimension` argument
        explicit_freq_dimension (ExplicitFrequencyDimension): TODO

    See Also:
        :class:`~zounds.spectral.FrequencyAdaptiveTransform`
    """
    def __new__(
            cls,
            arrs,
            time_dimension=None,
            scale=None,
            explicit_freq_dimension=None):

        if not time_dimension:
            raise ValueError('time_dimension is required')

        if explicit_freq_dimension:
            if scale:
                raise ValueError(
                    'scale must be None when explicit_freq_dimension is supplied')
            if not isinstance(arrs, np.ndarray):
                raise ValueError(
                    'arrs must be a contiguous array when explicit_freq_dimension_is_supplied')
            return ArrayWithUnits.__new__(
                cls, arrs, [time_dimension, explicit_freq_dimension])

        stops = list(np.cumsum([arr.shape[1] for arr in arrs]))
        slices = [slice(start, stop)
                  for (start, stop) in zip([0] + stops, stops)]
        dimensions = [time_dimension, ExplicitFrequencyDimension(scale, slices)]

        array = np.concatenate(arrs, axis=1)
        return ArrayWithUnits.__new__(cls, array, dimensions)

    def kwargs(self):
        return dict(
            time_dimension=self.time_dimension,
            explicit_freq_dimension=self.frequency_dimension)

    @property
    def scale(self):
        return self.frequency_dimension.scale

    @property
    def time_dimension(self):
        return self.dimensions[0]

    @property
    def frequency_dimension(self):
        return self.dimensions[1]

    @property
    def n_bands(self):
        return len(self.scale)

    def rasterize(self, n_coeffs):
        return self.square(n_coeffs)

    def square(self, n_coeffs, do_overlap_add=False):
        """
        Compute a "square" view of the frequency adaptive transform, by
        resampling each frequency band such that they all contain the same
        number of samples, and performing an overlap-add procedure in the
        case where the sample frequency and duration differ
        :param n_coeffs: The common size to which each frequency band should
        be resampled
        """
        resampled_bands = [
            resample(band, n_coeffs, axis=1).flatten()
            for band in self.iter_bands()]

        stacked = np.vstack(resampled_bands).T

        fdim = FrequencyDimension(self.scale)

        # TODO: This feels like it could be wrapped up nicely elsewhere
        chunk_frequency = Picoseconds(int(np.round(
            self.time_dimension.duration / Picoseconds(1) / n_coeffs)))

        td = TimeDimension(frequency=chunk_frequency)

        arr = ConstantRateTimeSeries(ArrayWithUnits(
            stacked.reshape(-1, n_coeffs, self.n_bands),
            dimensions=[self.time_dimension, td, fdim]))

        if not do_overlap_add:
            return arr

        # Begin the overlap add procedure
        overlap_ratio = self.time_dimension.overlap_ratio

        if overlap_ratio == 0:
            # no overlap add is necessary
            return ArrayWithUnits(stacked, [td, fdim])

        step_size_samples = int(n_coeffs * overlap_ratio)

        first_dim = int(np.round(
            (stacked.shape[0] * overlap_ratio) + (n_coeffs * overlap_ratio)))

        output = ArrayWithUnits(
            np.zeros((first_dim, self.n_bands)),
            dimensions=[td, fdim])

        for i, chunk in enumerate(arr):
            start = step_size_samples * i
            stop = start + n_coeffs
            output[start: stop] += chunk.reshape((-1, self.n_bands))

        return output

    def iter_bands(self):
        return (self[:, band] for band in self.scale)

    def like_dims(self, arr):
        return self.__class__(
            arr,
            time_dimension=self.time_dimension,
            explicit_freq_dimension=self.frequency_dimension)

    @classmethod
    def from_array_with_units(cls, arr):
        fdim = arr.dimensions[1]
        arrs = [arr[:, band] for band in fdim.scale]
        fa = FrequencyAdaptive(arrs, arr.dimensions[0], fdim.scale)
        return fa
