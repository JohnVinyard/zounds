import unittest2
from timeseries import TimeDimension
from duration import Seconds, Milliseconds


class TimeDimensionTests(unittest2.TestCase):
    def test_equals(self):
        td1 = TimeDimension(Seconds(1), Milliseconds(900))
        td2 = TimeDimension(Seconds(1), Milliseconds(900))
        self.assertEqual(td1, td2)

    def test_not_equal(self):
        td1 = TimeDimension(Seconds(2), Milliseconds(900))
        td2 = TimeDimension(Seconds(1), Milliseconds(900))
        self.assertNotEqual(td1, td2)
