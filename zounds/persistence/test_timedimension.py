import unittest2
from timedimension import TimeDimensionEncoder, TimeDimensionDecoder
from zounds.timeseries import TimeDimension, Seconds, Milliseconds


class TimeDimensionTests(unittest2.TestCase):
    def setUp(self):
        self.encoder = TimeDimensionEncoder()
        self.decoder = TimeDimensionDecoder()

    def test_can_round_trip(self):
        td = TimeDimension(Seconds(1), Milliseconds(500), 100)
        encoded = self.encoder.encode(td)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded, TimeDimension)
        self.assertEqual(Seconds(1), decoded.frequency)
        self.assertEqual(Milliseconds(500), decoded.duration)
        self.assertEqual(100, decoded.size)
