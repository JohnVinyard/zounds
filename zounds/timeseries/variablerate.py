import numpy as np
from timeseries import TimeSlice
from duration import Seconds


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
        try:
            start = self._data['timeslice'][0].start
            return TimeSlice(start=start, duration=self.end - start)
        except IndexError:
            return TimeSlice(start=Seconds(0), duration=Seconds(0))

    @property
    def end(self):
        try:
            return self._data['timeslice'][-1].end
        except IndexError:
            return Seconds(0)

    def __getitem__(self, index):
        if isinstance(index, TimeSlice):
            g = ((x['timeslice'], x['data']) for x in self._data
                 if x['timeslice'].end > index.start
                 and (index.duration is None or x['timeslice'].start < index.end))
            return VariableRateTimeSeries(g)
        if isinstance(index, int):
            return VariableRateTimeSeries((self._data[index],))
        else:
            return VariableRateTimeSeries(self._data[index])
