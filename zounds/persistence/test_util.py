import unittest2
from util import decode_timedelta
from zounds.timeseries import Seconds


class DecodeTimedeltaTests(unittest2.TestCase):
    def test_already_decoded_instance_is_returned(self):
        td = Seconds(10)
        decoded = decode_timedelta(td)
        self.assertEqual(td, decoded)
