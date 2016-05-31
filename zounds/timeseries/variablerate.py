import numpy as np
from timeseries import TimeSlice
from duration import Seconds, Picoseconds
from featureflow import Feature, BaseNumpyDecoder, NumpyEncoder


class VariableRateTimeSeries(object):
    def __init__(self, data):
        super(VariableRateTimeSeries, self).__init__()
        if isinstance(data, np.recarray):
            self._data = data
            return

        data = sorted(list(data), key=lambda x: x[0])
        try:
            example = data[0][1]
            shape = example.shape
            dtype = example.dtype
        except IndexError:
            shape = (0,)
            dtype = np.uint8
        self._data = np.recarray(len(data), dtype=[
            ('timeslice', TimeSlice),
            ('slicedata', dtype, shape)])
        self._data[:] = data

    def __len__(self):
        return self._data.__len__()

    @property
    def slices(self):
        return self._data.timeslice

    @property
    def slicedata(self):
        return self._data.slicedata

    @property
    def raw_data(self):
        return self._data

    @property
    def span(self):
        try:
            start = self._data.timeslice[0].start
            return TimeSlice(start=start, duration=self.end - start)
        except IndexError:
            return TimeSlice(duration=Seconds(0))

    @property
    def end(self):
        try:
            return self._data.timeslice[-1].end
        except IndexError:
            return Seconds(0)

    def __getitem__(self, index):
        if isinstance(index, TimeSlice):
            # TODO: Consider using a bisection approach here to make this much
            # faster than this brute-force, O(n) approach
            # compare the beginning of the index to the _end_ of each sample
            # compare the end of the index to the beginning of each sample
            g = ((x.timeslice, x.slicedata) for x in self._data
                 if x.timeslice.end > index.start and
                 (index.duration is None or x.timeslice.start < index.end))
            return VariableRateTimeSeries(g)
        elif isinstance(index, int):
            return self._data[index]
        else:
            return VariableRateTimeSeries(self._data[index])


class VariableRateTimeSeriesEncoder(NumpyEncoder):
    def __init__(self, needs=None):
        super(VariableRateTimeSeriesEncoder, self).__init__(needs=needs)

    def _prepare_data(self, data):
        output = np.recarray(len(data), dtype=[
            ('start', np.int64),
            ('duration', np.int64),
            ('slicedata', data.slicedata.dtype, data.slicedata.shape[1:])
        ])
        ps = Picoseconds(1)
        output.slicedata[:] = data.slicedata
        output.start[:] = [ts.start / ps for ts in data.slices]
        output.duration[:] = [ts.duration / ps for ts in data.slices]
        return output


class VariableRateTimeSeriesDecoder(BaseNumpyDecoder):
    def __init__(self):
        super(VariableRateTimeSeriesDecoder, self).__init__()

    def _gen(self, raw):
        for r in raw:
            ts = TimeSlice(
                    start=Picoseconds(r.start),
                    duration=Picoseconds(r.duration))
            yield (ts, r.slicedata)

    def _wrap_array(self, raw, metadata):
        return VariableRateTimeSeries(self._gen(raw))


class VariableRateTimeSeriesFeature(Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=VariableRateTimeSeriesEncoder,
            decoder=VariableRateTimeSeriesDecoder(),
            **extractor_args):
        super(VariableRateTimeSeriesFeature, self).__init__(
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
