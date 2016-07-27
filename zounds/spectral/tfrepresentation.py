from zounds.timeseries import ConstantRateTimeSeries
from frequencyscale import FrequencyBand


class TimeFrequencyRepresentation(ConstantRateTimeSeries):
    """
    A class that encapsulates time-frequency representation data.  The first
    axis represents time, the second axis represents frequency, and any
    subsequent axes contain multidimensional data about time-frequency positions
    """

    def __new__(cls, arr, frequency, duration=None, scale=None):
        if len(arr.shape) < 2:
            raise ValueError('arr must be at least 2D')

        if len(scale) != arr.shape[1]:
            raise ValueError('scale must have same size as dimension 2')

        obj = ConstantRateTimeSeries.__new__(cls, arr, frequency, duration)
        obj.scale = scale
        return obj

    def __array_finalize__(self, obj):
        super(TimeFrequencyRepresentation, self).__array_finalize__(obj)
        self.scale = getattr(obj, 'scale', None)

    def _freq_band_to_integer_indices(self, index):
        if not isinstance(index, FrequencyBand):
            return index

        return self.scale.get_slice(index)

    def __getitem__(self, index):
        try:
            slices = map(self._freq_band_to_integer_indices, index)
        except TypeError:
            slices = (self._freq_band_to_integer_indices(index),)
        return super(ConstantRateTimeSeries, self).__getitem__(slices)
