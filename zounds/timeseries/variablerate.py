import numpy as np
from timeseries import TimeSlice


class VariableRateTimeSeries(object):
    def __init__(self, data):
        super(VariableRateTimeSeries, self).__init__()
        if isinstance(data, np.recarray):
            self._data = data
            return

        data = list(data)
        try:
            example = data[0][1]
            shape = example.shape
            dtype = example.dtype
        except IndexError:
            shape = (0,)
            dtype = np.uint8
        self._data = np.array(data, dtype=[
                ('timeslice', TimeSlice),
                ('data', dtype, shape)])

    def __len__(self):
        return self._data.__len__()

    @property
    def span(self):
        start = self._data.timeslice[0].start
        return TimeSlice(start=start, duration=self.end-start)

    @property
    def end(self):
        return self._data.timeslice[-1].end

    def __getitem__(self, index):
        if isinstance(index, TimeSlice):
            raise NotImplementedError()
        return VariableRateTimeSeries(self._data[index])

