from featureflow import \
    NumpyEncoder, NumpyMetaData, Feature, BaseNumpyDecoder
import numpy as np
from duration import Picoseconds
from samplerate import SampleRate
import re
from zounds.core import Dimension


class TimeSlice(object):
    def __init__(self, duration=None, start=None):
        super(TimeSlice, self).__init__()

        if duration is not None and not isinstance(duration, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=duration.__class__))

        if start is not None and not isinstance(start, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=start.__class__))

        self.duration = duration
        self.start = start or Picoseconds(0)

    def __add__(self, other):
        return TimeSlice(self.duration, start=self.start + other)

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def end(self):
        return self.start + self.duration

    def __lt__(self, other):
        try:
            return self.start.__lt__(other.start)
        except AttributeError:
            return self.start.__lt__(other)

    def __gt__(self, other):
        try:
            return self.start.__gt__(other.start)
        except AttributeError:
            return self.start.__get__(other)

    def __le__(self, other):
        try:
            return self.start.__le__(other.start)
        except AttributeError:
            return self.start.__le__(other)

    def __ge__(self, other):
        try:
            return self.start.__ge__(other.start)
        except AttributeError:
            return self.start.__ge__(other)

    def __and__(self, other):
        delta = max(
                Picoseconds(0),
                min(self.end, other.end) - max(self.start, other.start))
        return TimeSlice(delta)

    def __contains__(self, other):
        print other
        if isinstance(other, np.timedelta64):
            return self.start < other < self.end
        if isinstance(other, TimeSlice):
            return other.start > self.start and other.end < self.end
        raise ValueError

    def __eq__(self, other):
        return self.start == other.start and self.duration == other.duration

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '{cls}(duration = {duration}, start = {start})'.format(
                cls=self.__class__.__name__,
                duration=str(self.duration),
                start=str(self.start))

    def __str__(self):
        return self.__repr__()


# class ConstantRateTimeSeriesMetadata(NumpyMetaData):
#     DTYPE_RE = re.compile(r'\[(?P<dtype>[^\]]+)\]')
#
#     def __init__(
#             self,
#             dtype=None,
#             shape=None,
#             frequency=None,
#             duration=None):
#         super(ConstantRateTimeSeriesMetadata, self).__init__(
#                 dtype=dtype, shape=shape)
#         self.frequency = self._decode_timedelta(frequency)
#         self.duration = self._decode_timedelta(duration)
#
#     @staticmethod
#     def from_timeseries(timeseries):
#         return ConstantRateTimeSeriesMetadata(
#                 dtype=timeseries.dtype,
#                 shape=timeseries.shape[1:],
#                 frequency=timeseries.frequency,
#                 duration=timeseries.duration)
#
#     def _encode_timedelta(self, td):
#         dtype = self.DTYPE_RE.search(str(td.dtype)).groupdict()['dtype']
#         return td.astype(np.uint64).tostring(), dtype
#
#     def _decode_timedelta(self, t):
#         if isinstance(t, np.timedelta64):
#             return t
#
#         v = np.fromstring(t[0], dtype=np.uint64)[0]
#         s = t[1]
#         return np.timedelta64(long(v), s)
#
#     def __repr__(self):
#         return repr((
#             str(np.dtype(self.dtype)),
#             self.shape,
#             self._encode_timedelta(self.frequency),
#             self._encode_timedelta(self.duration)
#         ))
#
#
# class BaseConstantRateTimeSeriesEncoder(NumpyEncoder):
#     def __init__(self, needs=None):
#         super(BaseConstantRateTimeSeriesEncoder, self).__init__(needs=needs)
#
#     def _prepare_data(self, data):
#         raise NotImplementedError()
#
#     def _prepare_metadata(self, data):
#         return ConstantRateTimeSeriesMetadata.from_timeseries(data)
#
#
# class ConstantRateTimeSeriesEncoder(BaseConstantRateTimeSeriesEncoder):
#     def __init__(self, needs=None):
#         super(ConstantRateTimeSeriesEncoder, self).__init__(needs=needs)
#
#     def _prepare_data(self, data):
#         return data
#
#
# class PackedConstantRateTimeSeriesEncoder(BaseConstantRateTimeSeriesEncoder):
#     def __init__(self, needs=None, axis=1):
#         super(PackedConstantRateTimeSeriesEncoder, self).__init__(needs=needs)
#         self.axis = axis
#
#     def _prepare_data(self, data):
#         packedbits = np.packbits(data.astype(np.uint8), axis=self.axis)
#
#         return ConstantRateTimeSeries(
#                 packedbits,
#                 frequency=data.frequency,
#                 duration=data.duration)
#
#
# class GreedyConstantRateTimeSeriesDecoder(BaseNumpyDecoder):
#     def __init__(self):
#         super(GreedyConstantRateTimeSeriesDecoder, self).__init__()
#
#     def _unpack_metadata(self, flo):
#         return ConstantRateTimeSeriesMetadata.unpack(flo)
#
#     def _wrap_array(self, raw, metadata):
#         return ConstantRateTimeSeries(
#                 raw, metadata.frequency, metadata.duration)
#
#
# class ConstantRateTimeSeriesFeature(Feature):
#     def __init__(
#             self,
#             extractor,
#             needs=None,
#             store=False,
#             key=None,
#             encoder=ConstantRateTimeSeriesEncoder,
#             decoder=GreedyConstantRateTimeSeriesDecoder(),
#             **extractor_args):
#         super(ConstantRateTimeSeriesFeature, self).__init__(
#                 extractor,
#                 needs=needs,
#                 store=store,
#                 encoder=encoder,
#                 decoder=decoder,
#                 key=key,
#                 **extractor_args)
#
#
class TimeDimension(Dimension):
    def __init__(self, frequency=None, duration=None, size=None):
        super(TimeDimension, self).__init__()
        self.size = size
        if not isinstance(frequency, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=frequency.__class__))

        if duration is not None and not isinstance(duration, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=duration.__class__))
        self.duration = duration or frequency
        self.frequency = frequency

    def __str__(self):
        fs = self.frequency / Picoseconds(int(1e12))
        ds = self.duration / Picoseconds(int(1e12))
        return 'TimeDimension(f={fs}, d={ds})'.format(**locals())

    def __repr__(self):
        return self.__str__()

    @property
    def samplerate(self):
        return SampleRate(self.frequency, self.duration)

    @property
    def overlap(self):
        return self.samplerate.overlap

    @property
    def duration_in_seconds(self):
        return self.duration / Picoseconds(int(1e12))

    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)

    @property
    def span(self):
        overlap = self.duration - self.frequency
        return TimeSlice((self.size * self.frequency) + overlap)

    @property
    def end(self):
        return self.span.end

    @property
    def end_seconds(self):
        return self.end / Picoseconds(int(1e12))

    def modified_dimension(self, size, windowsize, stepsize=None):
        stepsize = stepsize or windowsize
        yield TimeDimension(
                self.frequency * stepsize, self.duration * windowsize)
        yield self

    def metaslice(self, index, size):
        return TimeDimension(self.frequency, self.duration, size)

    def integer_based_slice(self, ts):
        if not isinstance(ts, TimeSlice):
            return ts

        diff = self.duration - self.frequency
        start_index = \
            max(0, np.floor((ts.start - diff) / self.frequency))
        end = self.end if ts.duration is None else ts.end

        # KLUDGE: This is basically arbitrary, but the motivation is that we'd
        # like to differentiate between cases where the slice
        # actually/intentionally overlaps a particular sample, and cases where
        # the slice overlaps the sample by a tiny amount, due to rounding or
        # lack of precision (e.g. Seconds(1) / SR44100().frequency).
        ratio = np.round(end / self.frequency, 2)

        stop_index = np.ceil(ratio)
        return slice(start_index, stop_index)

    def __eq__(self, other):
        return \
            self.frequency == other.frequency \
            and self.duration == other.duration