import unittest2
from api import \
    RangeRequest, RangeUnitUnsupportedException, ContentRange, ZoundsApp
from zounds.timeseries import TimeSlice, Seconds, Picoseconds, Milliseconds


class ContentRangeTests(unittest2.TestCase):
    def test_open_content_range(self):
        self.assertEqual(
                'bytes 10-100/100',
                str(ContentRange('bytes', 10, 100)))

    def test_closed_content_range(self):
        self.assertEqual(
                'seconds 10-90/100',
                str(ContentRange('seconds', 10, 100, stop=90)))

    def test_from_timeslce_full_slice(self):
        ts = TimeSlice()
        self.assertEqual(
                'seconds 0.0-100.0/100.0',
                str(ContentRange.from_timeslice(ts, Seconds(100))))

    def test_from_timeslice_open_ended(self):
        ts = TimeSlice(start=Picoseconds(int(1e12)) * 2.5)
        self.assertEqual(
                'seconds 2.5-100.0/100.0',
                str(ContentRange.from_timeslice(ts, Seconds(100))))

    def test_from_timeslice_closed(self):
        ts = TimeSlice(
                start=Picoseconds(int(1e12)) * 2.5,
                duration=Milliseconds(2000))
        self.assertEqual(
                'seconds 2.5-4.5/100.0',
                str(ContentRange.from_timeslice(ts, Seconds(100))))


class RangeRequestTests(unittest2.TestCase):
    def test_raises_for_unsupported_unit(self):
        rr = RangeRequest('hours=1-2')
        self.assertRaises(RangeUnitUnsupportedException, lambda: rr.range())

    def test_can_get_open_ended_byte_slice(self):
        rr = RangeRequest('bytes=10-')
        sl = rr.range()
        self.assertIsInstance(sl, slice)
        self.assertEqual((10, None, None), (sl.start, sl.stop, sl.step))

    def test_can_get_closed_byte_slice(self):
        rr = RangeRequest('bytes=10-100')
        sl = rr.range()
        self.assertIsInstance(sl, slice)
        self.assertEqual((10, 100, None), (sl.start, sl.stop, sl.step))

    def test_can_get_open_ended_time_slice(self):
        rr = RangeRequest('seconds=0-')
        sl = rr.range()
        self.assertIsInstance(sl, TimeSlice)
        self.assertEqual(TimeSlice(start=Seconds(0)), sl)

    def test_can_get_closed_time_slice(self):
        rr = RangeRequest('seconds=10.5-100.5')
        sl = rr.range()
        self.assertIsInstance(sl, TimeSlice)
        expected_start = Picoseconds(int(10.5 * 1e12))
        expected_duration = Picoseconds(int(90 * 1e12))
        self.assertEqual(
                TimeSlice(start=expected_start, duration=expected_duration), sl)


class ZoundsAppTests(unittest2.TestCase):
    def test_feature_paths_are_url_encoded(self):
        app = ZoundsApp(base_path='/zounds/')
        _id = 'http://example.com/resource'
        feature = 'bark'
        path = app.feature_path(_id, feature)
        expected = '/zounds/http%3A%2F%2Fexample.com%2Fresource/bark'
        self.assertEqual(expected, path)
