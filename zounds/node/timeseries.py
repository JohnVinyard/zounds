import numpy as np
import unittest2

'''
TODO: 
how do I handle "offset" in the ConstantRateTimeSeries class?
'''

class Seconds(np.timedelta64):
    
    def __new__(cls, seconds):
        return np.timedelta64(seconds, 's')

class Milliseconds(np.timedelta64):
    
    def __new__(cls, milliseconds):
        return np.timedelta64(milliseconds, 'ms')

class Microseconds(np.timedelta64):
    
    def __new__(self, microseconds):
        return np.timedelta64(microseconds, 'us')

class TimeSlice(object):
    
    def __init__(self, duration, start = None):
        super(TimeSlice, self).__init__()
        
        if not isinstance(duration, np.timedelta64):
            raise ValueError('duration must be of type {t}'.format(\
               t = np.timedelta64))
        
        if start != None and not isinstance(start, np.timedelta64):
            raise ValueError('start must be of type {t}'.format(\
               t = np.timedelta64))
        
        self.duration = duration
        self.start = start or np.timedelta64(0,'ns')
    
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
              duration = self.duration,
              start = self.start)
    
    def __str__(self):
        return self.__repr__()

class TimeSeries(object):
    '''
    TimeSeries interface
    '''
    def __init__(self):
        super(TimeSeries, self).__init__()

class ConstantRateTimeSeries(np.ndarray):
    '''
    A TimeSeries implementation with samples of a constant duration and 
    frequency.
    '''
    def __new__(cls, input_array, frequency, duration = None):
        
        if not isinstance(frequency, np.timedelta64):
            raise ValueError('duration must be of type {t}'.format(\
               t = np.timedelta64))
        
        if duration != None and not isinstance(duration, np.timedelta64):
            raise ValueError('start must be of type {t}'.format(\
               t = np.timedelta64))
            
        obj = np.asarray(input_array).view(cls)
        obj.frequency = frequency
        obj.duration = duration or frequency
        return obj
    
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
              ValueError, lambda : TimeSlice(np.timedelta64(100,'ns'),1))
    
    def test_can_instantiate_time_slice_instance_without_start_argument(self):
        duration = np.timedelta64(100,'s')
        ts = TimeSlice(duration)
        self.assertEqual(duration, ts.duration)
        self.assertEqual(np.timedelta64(0,'s'), ts.start)
        
    def test_can_instantiate_time_slice_instance_with_start_argument(self):
        duration = np.timedelta64(1000,'us')
        start = np.timedelta64(1,'h')
        ts = TimeSlice(duration, start = start)
        self.assertEqual(duration, ts.duration)
        self.assertEqual(start, ts.start)
    
    def test_can_intersect_two_time_slices(self):
        ts1 = TimeSlice(\
            np.timedelta64(100,'s'), start = np.timedelta64(100,'s'))
        ts2 = TimeSlice(\
            np.timedelta64(100,'s'), start = np.timedelta64(101,'s'))
        intersection = ts1 & ts2
        self.assertEqual(np.timedelta64(99,'s'), intersection.duration)
    
    def test_can_find_null_intersection(self):
        ts1 = TimeSlice(\
            np.timedelta64(100,'s'), start = np.timedelta64(100,'s'))
        ts2 = TimeSlice(\
            np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        intersection = ts1 & ts2
        self.assertEqual(np.timedelta64(0,'s'), intersection.duration)
    
    def test_does_not_contain_point_in_time_before(self):
        ts = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        self.assertFalse(np.timedelta64(10,'s') in ts)
    
    def test_contains_point_in_time_during(self):
        ts = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        self.assertTrue(np.timedelta64(210,'s') in ts)
    
    def test_does_not_contain_point_in_time_after(self):
        ts = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        self.assertFalse(np.timedelta64(310,'s') in ts)
    
    def test_does_not_contain_slice_completely_before(self):
        ts1 = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        ts2 = TimeSlice(np.timedelta64(10,'s'), np.timedelta64(12,'s'))
        self.assertFalse(ts2 in ts1)
    
    def test_does_not_contain_slice_beginning_before(self):
        ts1 = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        ts2 = TimeSlice(np.timedelta64(50,'s'), np.timedelta64(190,'s'))
        self.assertFalse(ts2 in ts1)
    
    def test_contains_slice(self):
        ts1 = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        ts2 = TimeSlice(np.timedelta64(10,'s'), np.timedelta64(250,'s'))
        self.assertTrue(ts2 in ts1)
    
    def test_does_not_contain_slice_completely_after(self):
        ts1 = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        ts2 = TimeSlice(np.timedelta64(100,'s'), np.timedelta64(310,'s'))
        self.assertFalse(ts2 in ts1)
    
    def test_does_not_contain_slice_beginning_after(self):
        ts1 = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
        ts2 = TimeSlice(np.timedelta64(100,'s'), np.timedelta64(210,'s'))
        self.assertFalse(ts2 in ts1)
    
    def test_raises_value_error_if_item_is_not_timedelta_or_timeslice(self):
        ts1 = TimeSlice(np.timedelta64(100,'s'), start = np.timedelta64(200,'s'))
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
        freq = np.timedelta64(1,'s')
        self.assertRaises(\
              ValueError, lambda : ConstantRateTimeSeries(arr, freq, 1))
    
    def test_duration_is_equal_to_frequency_if_not_provided(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(ts.frequency, ts.duration)
    
    def test_can_slice_time_series_with_time_slice(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(np.timedelta64(2,'s'),start = np.timedelta64(2,'s'))
        ts2 = ts[sl]
        self.assertEqual(2,len(ts2))
    
    def test_can_index_constant_rate_time_series_with_integer_index(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        ts2 = ts[5]
        self.assertEqual(5, ts2)
    
    def test_can_slice_constant_rate_time_series_with_integer_indices(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        ts2 = ts[:5]
        self.assertEqual(5,len(ts2))
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
    
    def test_can_add_constant_factor_to_time_series(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        ts2 = ts + 10
        self.assertTrue(np.all(np.arange(10,20) == ts2))
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
    
    def test_get_index_error_when_using_out_of_range_int_index(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertRaises(IndexError, lambda : ts[100])
    
    def test_get_empty_time_series_when_using_out_of_range_time_slice(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(np.timedelta64(2,'s'), start = np.timedelta64(11,'s'))
        ts2 = ts[sl]
        self.assertEqual(0, ts2.size)
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
    
    def test_time_slice_spanning_less_than_one_sample_returns_one_sample(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(\
           np.timedelta64(100,'ms'), start = np.timedelta64(1500,'ms'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(1, ts2.size)
        self.assertEqual(1, ts2[0])
    
    def test_time_slice_spanning_multiple_samples_returns_all_samples(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(\
           np.timedelta64(2000,'ms'), start = np.timedelta64(1500,'ms'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(3, ts2.size)
        self.assertEqual(1, ts2[0])
        self.assertEqual(2, ts2[1])
        self.assertEqual(3, ts2[2])
    
    def test_frequency_and_duration_differ(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        duration = np.timedelta64(2,'s')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(np.timedelta64(2, 's'), start = np.timedelta64(1,'s'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(3, ts2.size)
        self.assertEqual(0, ts2[0])
        self.assertEqual(1, ts2[1])
        self.assertEqual(2, ts2[2])
    
    def test_frequency_and_duration_differ2(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        duration = np.timedelta64(3,'s')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(np.timedelta64(2, 's'), start = np.timedelta64(5,'s'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertEqual(3, ts2[0])
        self.assertEqual(4, ts2[1])
        self.assertEqual(5, ts2[2])
        self.assertEqual(6, ts2[3])
    
    def test_frequency_and_duration_differ3(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        duration = np.timedelta64(3,'s')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(np.timedelta64(2, 's'), start = np.timedelta64(6,'s'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertEqual(4, ts2[0])
        self.assertEqual(5, ts2[1])
        self.assertEqual(6, ts2[2])
        self.assertEqual(7, ts2[3])
    
    def test_frequency_less_than_one(self):
        arr = np.arange(10)
        freq = np.timedelta64(500,'ms')
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(np.timedelta64(2, 's'), start = np.timedelta64(600,'ms'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(5, ts2.size)
        self.assertTrue(np.all(np.arange(1,6) == ts2))
    
    def test_frequency_less_than_one_freq_and_duration_differ(self):
        arr = np.arange(10)
        freq = np.timedelta64(500,'ms')
        duration = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(np.timedelta64(3, 's'), start = np.timedelta64(250, 'ms'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(7, ts2.size)
        self.assertTrue(np.all(np.arange(0,7) == ts2))
    
    def test_frequency_less_than_one_freq_and_duration_differ2(self):
        arr = np.arange(10)
        freq = np.timedelta64(500,'ms')
        duration = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(np.timedelta64(3, 's'), start = np.timedelta64(1250, 'ms'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(8, ts2.size)
        self.assertTrue(np.all(np.arange(1,9) == ts2))
    
    def test_duration_less_than_frequency(self):
        arr = np.arange(10)
        freq = np.timedelta64(1, 's')
        duration = np.timedelta64(500, 'ms')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(np.timedelta64(3,'s'), start = np.timedelta64(1250, 'ms'))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(1,5) == ts2))
        
    def test_span_freq_and_duration_equal(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(TimeSlice(np.timedelta64(10,'s')), ts.span)
    
    def test_span_duration_greater_than_frequency(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        duration = np.timedelta64(2500,'ms')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        self.assertEqual(TimeSlice(np.timedelta64(11500,'ms')), ts.span)
    
    def test_span_duration_less_than_frequency(self):
        arr = np.arange(10)
        freq = np.timedelta64(1,'s')
        duration = np.timedelta64(500,'ms')
        ts = ConstantRateTimeSeries(arr, freq, duration)
        self.assertEqual(TimeSlice(np.timedelta64(9500,'ms')), ts.span)