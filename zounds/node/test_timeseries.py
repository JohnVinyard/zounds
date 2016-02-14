import numpy as np
import unittest2
from duration import \
    Picoseconds, Milliseconds, Seconds, Microseconds, Nanoseconds, Hours
from timeseries import TimeSlice, ConstantRateTimeSeries


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
        self.assertRaises(ValueError, lambda: TimeSlice(1))

    def test_raises_if_start_is_provided_and_is_not_timedelta_instance(self):
        self.assertRaises( \
                ValueError, lambda: TimeSlice(Nanoseconds(100), 1))

    def test_can_instantiate_time_slice_instance_without_start_argument(self):
        duration = Seconds(100)
        ts = TimeSlice(duration)
        self.assertEqual(duration, ts.duration)
        self.assertEqual(Seconds(0), ts.start)

    def test_can_instantiate_time_slice_instance_with_start_argument(self):
        duration = Microseconds(1000)
        start = Hours(1)
        ts = TimeSlice(duration, start=start)
        self.assertEqual(duration, ts.duration)
        self.assertEqual(start, ts.start)

    def test_can_intersect_two_time_slices(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(100))
        ts2 = TimeSlice(Seconds(100), start=Seconds(101))
        intersection = ts1 & ts2
        self.assertEqual(Seconds(99), intersection.duration)

    def test_can_find_null_intersection(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(100))
        ts2 = TimeSlice(Seconds(100), start=Seconds(200))
        intersection = ts1 & ts2
        self.assertEqual(Seconds(0), intersection.duration)

    def test_does_not_contain_point_in_time_before(self):
        ts = TimeSlice(Seconds(100), start=Seconds(200))
        self.assertFalse(Seconds(10) in ts)

    def test_contains_point_in_time_during(self):
        ts = TimeSlice(Seconds(100), start=Seconds(200))
        self.assertTrue(Seconds(210) in ts)

    def test_does_not_contain_point_in_time_after(self):
        ts = TimeSlice(Seconds(100), start=Seconds(200))
        self.assertFalse(Seconds(310) in ts)

    def test_does_not_contain_slice_completely_before(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(200))
        ts2 = TimeSlice(Seconds(10), Seconds(12))
        self.assertFalse(ts2 in ts1)

    def test_does_not_contain_slice_beginning_before(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(200))
        ts2 = TimeSlice(Seconds(50), Seconds(190))
        self.assertFalse(ts2 in ts1)

    def test_contains_slice(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(200))
        ts2 = TimeSlice(Seconds(10), Seconds(250))
        self.assertTrue(ts2 in ts1)

    def test_does_not_contain_slice_completely_after(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(200))
        ts2 = TimeSlice(Seconds(100), Seconds(310))
        self.assertFalse(ts2 in ts1)

    def test_does_not_contain_slice_beginning_after(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(200))
        ts2 = TimeSlice(Seconds(100), Seconds(210))
        self.assertFalse(ts2 in ts1)

    def test_raises_value_error_if_item_is_not_timedelta_or_timeslice(self):
        ts1 = TimeSlice(Seconds(100), start=Seconds(200))
        self.assertRaises(ValueError, lambda: 's' in ts1)

    def test_eq_when_start_and_duration_equal(self):
        ts1 = TimeSlice(Seconds(2), start=Seconds(2))
        ts2 = TimeSlice(Seconds(2), start=Seconds(2))
        self.assertEqual(ts1, ts2)

    def test_ne_when_durations_differ(self):
        ts1 = TimeSlice(Seconds(2), start=Seconds(2))
        ts2 = TimeSlice(Seconds(3), start=Seconds(2))
        self.assertNotEqual(ts1, ts2)

    def test_ne_when_starts_differ(self):
        ts1 = TimeSlice(Seconds(2), start=Seconds(2))
        ts2 = TimeSlice(Seconds(2), start=Seconds(3))
        self.assertNotEqual(ts1, ts2)


class TimeSeriesTests(unittest2.TestCase):
    def test_raises_if_frequency_is_not_timedelta_instance(self):
        arr = np.arange(10)
        self.assertRaises(ValueError, lambda: ConstantRateTimeSeries(arr, 1))

    def test_raises_if_duration_is_not_timedelta_instance(self):
        arr = np.arange(10)
        freq = Seconds(1)
        self.assertRaises( \
                ValueError, lambda: ConstantRateTimeSeries(arr, freq, 1))

    def test_duration_is_equal_to_frequency_if_not_provided(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(ts.frequency, ts.duration)

    def test_can_slice_time_series_with_time_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Seconds(2), start=Seconds(2))
        ts2 = ts[sl]
        self.assertEqual(2, len(ts2))

    def test_can_slice_time_series_with_open_ended_time_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(None, start=Seconds(2))
        ts2 = ts[sl]
        self.assertEqual(8, len(ts2))

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
        self.assertEqual(5, len(ts2))
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
        self.assertRaises(IndexError, lambda: ts[100])

    def test_get_empty_time_series_when_using_out_of_range_time_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Seconds(2), start=Seconds(11))
        ts2 = ts[sl]
        self.assertEqual(0, ts2.size)
        self.assertIsInstance(ts2, ConstantRateTimeSeries)

    def test_time_slice_spanning_less_than_one_sample_returns_one_sample(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Milliseconds(100), start=Milliseconds(1500))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(1, ts2.size)
        self.assertEqual(1, ts2[0])

    def test_time_slice_spanning_multiple_samples_returns_all_samples(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Milliseconds(2000), start=Milliseconds(1500))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(3, ts2.size)
        self.assertTrue(np.all(np.arange(1, 4) == ts2))

    def test_frequency_and_duration_differ(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Seconds(2)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(2), start=Seconds(1))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(3, ts2.size)
        self.assertTrue(np.all(np.arange(3) == ts2))

    def test_frequency_and_duration_differ2(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Seconds(3)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(2), start=Seconds(5))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(3, 7) == ts2))

    def test_frequency_and_duration_differ3(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Seconds(3)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(2), start=Seconds(6))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(4, 8) == ts2))

    def test_frequency_less_than_one(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice(Seconds(2), start=Milliseconds(600))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(5, ts2.size)
        self.assertTrue(np.all(np.arange(1, 6) == ts2))

    def test_frequency_less_than_one_freq_and_duration_differ(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        duration = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(3), start=Milliseconds(250))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(7, ts2.size)
        self.assertTrue(np.all(np.arange(0, 7) == ts2))

    def test_frequency_less_than_one_freq_and_duration_differ2(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        duration = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(3), start=Milliseconds(1250))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(8, ts2.size)
        self.assertTrue(np.all(np.arange(1, 9) == ts2))

    def test_duration_less_than_frequency(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        sl = TimeSlice(Seconds(3), start=Milliseconds(1250))
        ts2 = ts[sl]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertEqual(4, ts2.size)
        self.assertTrue(np.all(np.arange(1, 5) == ts2))

    def test_can_get_entire_time_series_with_empty_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        duration = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq, duration)
        ts2 = ts[:]
        self.assertIsInstance(ts2, ConstantRateTimeSeries)
        self.assertTrue(np.all(np.arange(10) == ts2))

    def test_can_get_entire_time_series_with_empty_time_slice(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        sl = TimeSlice()
        ts2 = ts[sl]
        self.assertEqual(10, len(ts2))

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

    def test_duration_in_seconds_half_second(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(0.5, ts.duration_in_seconds)

    def test_duration_in_seconds_two_seconds(self):
        arr = np.arange(10)
        freq = Seconds(2)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(2, ts.duration_in_seconds)

    def test_samplerate_one_per_second(self):
        arr = np.arange(10)
        freq = Seconds(1)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(1, ts.samples_per_second)

    def test_samplerate_two_per_second(self):
        arr = np.arange(10)
        freq = Milliseconds(500)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(2, ts.samples_per_second)

    def test_samplerate_three_per_second(self):
        arr = np.arange(10)
        freq = Milliseconds(333)
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(3, int(ts.samples_per_second))

    def test_samplerate_audio(self):
        arr = np.arange(10)
        freq = Picoseconds(int(1e12)) / 44100.
        ts = ConstantRateTimeSeries(arr, freq)
        self.assertEqual(44100, int(ts.samples_per_second))

    def test_concatenation_with_differing_freqs_and_durations_raises(self):
        ts = ConstantRateTimeSeries( \
                np.arange(10),
                Seconds(1),
                Seconds(2))
        ts2 = ConstantRateTimeSeries( \
                np.arange(10, 20),
                Seconds(1),
                Seconds(1))
        self.assertRaises(ValueError, lambda: ts.concatenate(ts2))

    def test_concatenation_with_matching_freqs_and_duration_results_in_crts(
            self):
        ts = ConstantRateTimeSeries( \
                np.ones((10, 3)),
                Seconds(1),
                Seconds(2))
        ts2 = ConstantRateTimeSeries( \
                np.ones((13, 3)),
                Seconds(1),
                Seconds(2))
        result = ts.concatenate(ts2)
        self.assertIsInstance(result, ConstantRateTimeSeries)
        self.assertEqual((23, 3), result.shape)

    def test_concat_with_differing_freqs(self):
        ts = ConstantRateTimeSeries( \
                np.ones((10, 3)),
                Seconds(2),
                Seconds(2))
        ts2 = ConstantRateTimeSeries( \
                np.ones((13, 3)),
                Seconds(1),
                Seconds(2))
        self.assertRaises( \
                ValueError, lambda: ConstantRateTimeSeries.concat([ts, ts2]))

    def test_concat_with_differing_durations(self):
        ts = ConstantRateTimeSeries( \
                np.ones((10, 3)),
                Seconds(1),
                Seconds(2))
        ts2 = ConstantRateTimeSeries( \
                np.ones((13, 3)),
                Seconds(1),
                Seconds(3))
        self.assertRaises( \
                ValueError, lambda: ConstantRateTimeSeries.concat([ts, ts2]))

    def test_concat_along_first_axis(self):
        ts = ConstantRateTimeSeries( \
                np.ones((10, 3)),
                Seconds(1),
                Seconds(2))
        ts2 = ConstantRateTimeSeries( \
                np.ones((13, 3)),
                Seconds(1),
                Seconds(2))
        result = ConstantRateTimeSeries.concat([ts, ts2])
        self.assertEqual((23, 3), result.shape)

    def test_concat_along_second_axis(self):
        ts = ConstantRateTimeSeries( \
                np.ones((10, 3)),
                Seconds(1),
                Seconds(2))
        ts2 = ConstantRateTimeSeries( \
                np.ones((10, 5)),
                Seconds(1),
                Seconds(2))
        result = ConstantRateTimeSeries.concat([ts, ts2], axis=1)
        self.assertEqual((10, 8), result.shape)
