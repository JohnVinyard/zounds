from zounds.core import ArrayWithUnits
from timeseries import TimeDimension, TimeSlice


class ConstantRateTimeSeries(ArrayWithUnits):
    def __new__(cls, array):
        try:
            dim = array.dimensions[0]
        except AttributeError:
            raise ValueError('array must be of type ArrayWithUnits')

        if not isinstance(dim, TimeDimension):
            raise ValueError('array first dimension must be a TimeDimension')

        return ArrayWithUnits.__new__(cls, array, array.dimensions)

    def iter_slices(self):
        td = self.dimensions[0]
        for i, data in enumerate(self):
            yield TimeSlice(duration=td.duration, start=td.frequency * i), data
