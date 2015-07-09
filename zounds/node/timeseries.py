import unittest2
from flow import NumpyMetaData, Node, Feature, Decoder
import numpy as np
from duration import \
    Picoseconds, Milliseconds, Seconds, Microseconds, Nanoseconds, Hours
from samplerate import SampleRate

class TimeSlice(object):
    
    def __init__(self, duration, start = None):
        super(TimeSlice, self).__init__()
        
        if not isinstance(duration, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = duration.__class__))
        
        if start != None and not isinstance(start, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = start.__class__))
        
        self.duration = duration
        self.start = start or np.timedelta64(0,'s')
    
    def __add__(self, other):
        return TimeSlice(self.duration, start = self.start + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    @property
    def end(self):
        return self.start + self.duration
    
    def __and__(self, other):
        delta = max(\
           np.timedelta64(0,'s'),
           min(self.end, other.end) - max(self.start, other.start))
        return TimeSlice(delta)

    def __contains__(self, other):
        if isinstance(other, np.timedelta64):
            return other > self.start and other < self.end
        if isinstance(other, TimeSlice):
            return other.start > self.start and other.end < self.end
        raise ValueError
    
    def __eq__(self, other):
        return self.start == other.start and self.duration == other.duration
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '{cls}(duration = {duration}, start = {start})'.format(\
              cls = self.__class__.__name__,
              duration = str(self.duration),
              start = str(self.start))
    
    def __str__(self):
        return self.__repr__()

class TimeSeries(object):
    '''
    TimeSeries interface
    '''
    def __init__(self):
        super(TimeSeries, self).__init__()

class ConstantRateTimeSeriesMetadata(NumpyMetaData):
    
    def __init__(\
         self, 
         dtype = None, 
         shape = None, 
         frequency = None, 
         duration = None):
        
        super(ConstantRateTimeSeriesMetadata, self).__init__(\
            dtype = dtype, shape = shape)
        self.frequency = self._decode_timedelta(frequency)
        self.duration = self._decode_timedelta(duration)
    
    def _encode_timedelta(self, td):
        return (td.astype(np.uint64).tostring(), str(td.dtype)[-3:-1])

    def _decode_timedelta(self, t):
        if isinstance(t, np.timedelta64):
            return t
        
        v = np.fromstring(t[0], dtype = np.uint64)[0]
        s = t[1] 
        return np.timedelta64(long(v), s)
    
    def __repr__(self):
        return repr((
             str(np.dtype(self.dtype)), 
             self.shape,
             self._encode_timedelta(self.frequency),
             self._encode_timedelta(self.duration)
         ))

class ConstantRateTimeSeriesEncoder(Node):
    
    content_type = 'application/octet-stream'
    
    def __init__(self, needs = None):
        super(ConstantRateTimeSeriesEncoder, self).__init__(needs = needs)
        self.metadata = None
    
    
    def _process(self, data):
        if not self.metadata:
            self.metadata = ConstantRateTimeSeriesMetadata(\
                dtype = data.dtype, 
                shape = data.shape[1:],
                frequency = data.frequency,
                duration = data.duration)
            packed = self.metadata.pack()
            yield packed
        
        encoded = data.tostring()
        yield encoded

def _np_from_buffer(b, shape, dtype, freq, duration):
    f = np.frombuffer if len(b) else np.fromstring
    shape = tuple(int(x) for x in shape)
    f = f(b, dtype = dtype).reshape(shape)
    return ConstantRateTimeSeries(f, freq, duration)

class GreedyConstantRateTimeSeriesDecoder(Decoder):
    
    def __init__(self):
        super(GreedyConstantRateTimeSeriesDecoder, self).__init__()

    def __call__(self,flo):
        metadata, bytes_read = ConstantRateTimeSeriesMetadata.unpack(flo)
        
        leftovers = flo.read()
        leftover_bytes = len(leftovers)
        first_dim = leftover_bytes / metadata.totalsize
        dim = (first_dim,) + metadata.shape
        out = _np_from_buffer(\
           leftovers,
           dim,
           metadata.dtype, 
           metadata.frequency, 
           metadata.duration)
        return out

    def __iter__(self,flo):
        yield self(flo)

class ConstantRateTimeSeriesFeature(Feature):
    
    def __init__(\
        self,
        extractor,
        needs = None,
        store = False,
        key = None,
        decoder = GreedyConstantRateTimeSeriesDecoder(),
        **extractor_args):
        
        super(ConstantRateTimeSeriesFeature ,self).__init__(\
            extractor,
            needs = needs,
            store = store,
            encoder = ConstantRateTimeSeriesEncoder,
            decoder = decoder,
            key = key,
            **extractor_args)
        
class ConstantRateTimeSeries(np.ndarray):
    '''
    A TimeSeries implementation with samples of a constant duration and 
    frequency.
    '''
    __array_priority__ = 10.0
    
    def __new__(cls, input_array, frequency, duration = None):
        
        if not isinstance(frequency, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = frequency.__class__))
        
        if duration != None and not isinstance(duration, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = duration.__class__))
        
        obj = np.asarray(input_array).view(cls)
        obj.frequency = frequency
        obj.duration = duration or frequency
        return obj

    def concatenate(self, other):
        if self.frequency == other.frequency and self.duration == other.duration:
            return ConstantRateTimeSeries(\
                  np.concatenate([self, other]),
                  frequency = self.frequency,
                  duration = self.duration)
        raise ValueError(\
         'self and other must have the same sample frequency and sample duration')

    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)
    
    @property
    def samplerate(self):
        return SampleRate(self.frequency, self.duration)
    
    @property
    def overlap(self):
        return self.samplerate.overlap
    
    @property
    def span(self):
        overlap = self.duration - self.frequency
        return TimeSlice((len(self) * self.frequency) + overlap) 

    def __array_finalize__(self, obj):
        if obj is None: return
        self.frequency = getattr(obj, 'frequency', None)
        self.duration = getattr(obj, 'duration', None)
        
    def __getitem__(self, index):
        if isinstance(index, TimeSlice):
            diff = self.duration - self.frequency
            start_index = \
                max(0, np.floor((index.start - diff) / self.frequency))
            stop_index = np.ceil(index.end / self.frequency)
            return self[start_index : stop_index]
        
        return super(ConstantRateTimeSeries, self).__getitem__(index)

class ConvenienceClassTests(unittest2.TestCase):
    
    def test_seconds_equal(self):
        a = Seconds(1)
        b = np.timedelta64(1, 's')
        self.assertEqual(a, b)
    
    def test_milliseconds_equal(self):
        a = Milliseconds(1000)
        b = np.timedelta64(1000, 'ms')
        self.assertEqual(a, b)
    
    def test_microseconds_equal(self):
        a = Microseconds(10)
        b = np.timedelta64(10, 'us')
        self.assertEqual(a, b)
    
    def test_different_units_equal(self):
        self.assertEqual(Seconds(1), Milliseconds(1000))

class TimeSliceTests(unittest2.TestCase):
    
    def test_raises_if_duration_is_not_timedelta_instance(self):
        self.assertRaises(ValueError, lambda : TimeSlice(1))
    
    def test_raises_if_start_is_provided_and_is_not_timedelta_instance(self):
        self.assertRaises(\
              ValueError, lambda : TimeSlice(Nanoseconds(100), 1))
    
    def test_can_instantiate_time_slice_instance_without_start_argument(self):
        duration = Seconds(100)
        ts = TimeSlice(duration)
        self.assertEqual(duration, ts.duration)
        self.assertEqual(Seconds(0), ts.start)
        
    def test_can_instantiate_time_slice_instance_with_start_argument(self):
        duration = Microseconds(1000)
        start = Hours(1)
        ts = TimeSlice(duration, start = start)
        self.assertEqual(duration, ts.duration)
        self.assertEqual(start, ts.start)
    
    def test_can_intersect_two_time_slices(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(100))
        ts2 = TimeSlice(Seconds(100), start = Seconds(101))
        intersection = ts1 & ts2
        self.assertEqual(Seconds(99), intersection.duration)
    
    def test_can_find_null_intersection(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(100))
        ts2 = TimeSlice(Seconds(100), start = Seconds(200))
        intersection = ts1 & ts2
        self.assertEqual(Seconds(0), intersection.duration)
    
    def test_does_not_contain_point_in_time_before(self):
        ts = TimeSlice(Seconds(100), start = Seconds(200))
        self.assertFalse(Seconds(10) in ts)
    
    def test_contains_point_in_time_during(self):
        ts = TimeSlice(Seconds(100), start = Seconds(200))
        self.assertTrue(Seconds(210) in ts)
    
    def test_does_not_contain_point_in_time_after(self):
        ts = TimeSlice(Seconds(100), start = Seconds(200))
        self.assertFalse(Seconds(310) in ts)
    
    def test_does_not_contain_slice_completely_before(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(200))
        ts2 = TimeSlice(Seconds(10), Seconds(12))
        self.assertFalse(ts2 in ts1)
    
    def test_does_not_contain_slice_beginning_before(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(200))
        ts2 = TimeSlice(Seconds(50), Seconds(190))
        self.assertFalse(ts2 in ts1)
    
    def test_contains_slice(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(200))
        ts2 = TimeSlice(Seconds(10), Seconds(250))
        self.assertTrue(ts2 in ts1)
    
    def test_does_not_contain_slice_completely_after(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(200))
        ts2 = TimeSlice(Seconds(100), Seconds(310))
        self.assertFalse(ts2 in ts1)
    
    def test_does_not_contain_slice_beginning_after(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(200))
        ts2 = TimeSlice(Seconds(100), Seconds(210))
        self.assertFalse(ts2 in ts1)
    
    def test_raises_value_error_if_item_is_not_timedelta_or_timeslice(self):
        ts1 = TimeSlice(Seconds(100), start = Seconds(200))
        self.assertRaises(ValueError, lambda : 's' in ts1)
    
    def test_eq_when_start_and_duration_equal(self):
        ts1 = TimeSlice(Seconds(2), start = Seconds(2))
        ts2 = TimeSlice(Seconds(2), start = Seconds(2))
        self.assertEqual(ts1, ts2)
    
    def test_ne_when_durations_differ(self):
        ts1 = TimeSlice(Seconds(2), start = Seconds(2))
        ts2 = TimeSlice(Seconds(3), start = Seconds(2))
        self.assertNotEqual(ts1, ts2)
    
    def test_ne_when_starts_differ(self):
        ts1 = TimeSlice(Seconds(2), start = Seconds(2))
        ts2 = TimeSlice(Seconds(2), start = Seconds(3))
        self.assertNotEqual(ts1, ts2)
    
class TimeSeriesTests(unittest2.TestCase):
    
    def test_raises_if_frequency_is_not_timedelta_instance(self):
        arr = np.arange(10)
        self.assertRaises(ValueError, lambda: ConstantRateTimeSeries(arr, 1))
    
    def test_raises_if_duration_is_not_timedelta_instance(self):
        arr = np.arange(10)
        freq = Seconds(1)
        self.assertRaises(\
              ValueError, lambda : ConstantRateTimeSeries(arr, freq, 1))
    
    def test_duration_is_equal_to_frequency_if_not_provided(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(ts.frequency, ts.duration)
    
    def test_can_slice_time_series_with_time_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Seconds(2), start = Seconds(2))
        ts2 = ts[sl]
        self.assertEqual(2,len(ts2))
    
    def test_can_index_constant_rate_time_series_with_integer_index(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        ts2 = ts[5]
        self.assertEqual(5, ts2)
    
    def test_can_slice_constant_rate_time_series_with_integer_indices(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        ts2 = ts[:5]
        self.assertEqual(5,len(ts2))
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
    
    def test_can_add_constant_factor_to_time_series(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        ts2 = ts + 10
        self.assertTrue(np.all(np.arange(10, 20) == ts2))
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
    
    def test_get_index_error_when_using_out_of_range_int_index(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertRaises(IndexError, lambda : ts[100])
    
    def test_get_empty_time_series_when_using_out_of_range_time_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Seconds(2), start = Seconds(11))
        ts2 = ts[sl]
        self.assertEqual(0, ts2.size)
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
    
    def test_time_slice_spanning_less_than_one_sample_returns_one_sample(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Milliseconds(100), start = Milliseconds(1500))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(1, ts2.size)
        self.assertEqual(1, ts2[0])
    
    def test_time_slice_spanning_multiple_samples_returns_all_samples(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Milliseconds(2000), start = Milliseconds(1500))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(3, ts2.size)
        self.assertTrue(np.all(np.arange(1,4) == ts2))
    
    def test_frequency_and_duration_differ(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Seconds(2)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(2), start = Seconds(1))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(3, ts2.size)
        self.assertTrue(np.all(np.arange(3) == ts2))
    
    def test_frequency_and_duration_differ2(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Seconds(3)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(2), start = Seconds(5))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(3,7) == ts2))
    
    def test_frequency_and_duration_differ3(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Seconds(3)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(2), start = Seconds(6))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(4,8) == ts2))
    
    def test_frequency_less_than_one(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Seconds(2), start = Milliseconds(600))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(5, ts2.size)
        self.assertTrue(np.all(np.arange(1,6) == ts2))
    
    def test_frequency_less_than_one_freq_and_duration_differ(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        duration = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(3), start = Milliseconds(250))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(7, ts2.size)
        self.assertTrue(np.all(np.arange(0,7) == ts2))
    
    def test_frequency_less_than_one_freq_and_duration_differ2(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        duration = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(3), start = Milliseconds(1250))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(8, ts2.size)
        self.assertTrue(np.all(np.arange(1,9) == ts2))
    
    def test_duration_less_than_frequency(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(3), start = Milliseconds(1250))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(1,5) == ts2))
    
    def test_can_get_entire_time_series_with_empty_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        ts2 = ts[:]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertTrue(np.all(np.arange(10) == ts2))
        
    def test_span_freq_and_duration_equal(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(TimeSlice(Seconds(10)), ts.span)
    
    def test_span_duration_greater_than_frequency(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Milliseconds(2500)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        self.assertEqual(TimeSlice(Milliseconds(11500)), ts.span)
    
    def test_span_duration_less_than_frequency(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        self.assertEqual(TimeSlice(Milliseconds(9500)), ts.span)
    
    def test_samplerate_one_per_second(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(1, ts.samplerate())
    
    def test_samplerate_two_per_second(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(2, ts.samplerate())
    
    def test_samplerate_three_per_second(self):
        arr = np.arange(10)
        freq = Milliseconds(333)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(3, int(ts.samplerate()))
    
    def test_samplerate_audio(self):
        arr = np.arange(10)
        freq = Picoseconds(int(1e12)) / 44100.
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(44100, int(ts.samplerate()))
    
    def test_concatenation_with_differing_freqs_and_durations_raises(self):
        ts = ConstantRateTimeSeries(\
            np.arange(10),
            Seconds(1),
            Seconds(2))
        ts2 = ConstantRateTimeSeries(\
             np.arange(10, 20),
             Seconds(1),
             Seconds(1)) 
        self.assertRaises(ValueError, lambda : ts.concatenate(ts2))
        
    def test_concatenation_with_matching_freqs_and_duration_results_in_crts(self):
        ts = ConstantRateTimeSeries(\
            np.ones((10, 3)),
            Seconds(1),
            Seconds(2))
        ts2 = ConstantRateTimeSeries(\
             np.ones((13, 3)),
             Seconds(1),
             Seconds(2))
        result = ts.concatenate(ts2)
        self.assertIsInstance(result, ConstantRateTimeSeries)
        self.assertEqual((23, 3), result.shape)
        