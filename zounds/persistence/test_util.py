import unittest2
from util import decode_timedelta, extract_init_args
from zounds.timeseries import Seconds


class DecodeTimedeltaTests(unittest2.TestCase):
    def test_already_decoded_instance_is_returned(self):
        td = Seconds(10)
        decoded = decode_timedelta(td)
        self.assertEqual(td, decoded)


class ExtractInitArgsTests(unittest2.TestCase):
    def test_should_not_include_locals(self):
        class Blah(object):
            def __init__(self, x, y):
                super(Blah, self).__init__()
                self.x = x
                z = 10
                self.y = y + z

        b = Blah(20, 30)
        args = extract_init_args(b)
        self.assertEqual(2, len(args))
        self.assertIn(20, args)
        self.assertIn(40, args)
        self.assertNotIn(10, args)
