import unittest2
from timeslice import TimeSliceEncoder, TimeSliceDecoder
from zounds.timeseries import TimeSlice, Seconds, Milliseconds


class TimeSliceRoundTripTests(unittest2.TestCase):
    def _roundtrip(self, ts):
        encoder = TimeSliceEncoder()
        decoder = TimeSliceDecoder()
        encoded = encoder.dict(ts)
        decoded = TimeSlice(**decoder.kwargs(encoded))
        self.assertEqual(ts, decoded)

    def test_can_roundtrip(self):
        self._roundtrip(TimeSlice(start=Seconds(1), duration=Seconds(1)))

    def test_can_roundtrip_milliseconds(self):
        self._roundtrip(TimeSlice(start=Seconds(1), duration=Milliseconds(250)))
