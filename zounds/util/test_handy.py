import unittest2
from handy import tuplify


class TuplifyTests(unittest2.TestCase):
    def test_tuple_from_tuple(self):
        self.assertEqual((1, 2, 3), tuplify((1, 2, 3)))

    def test_tuple_from_list(self):
        self.assertEqual((1, 2, 3), tuplify([1, 2, 3]))

    def test_tuple_from_integer(self):
        self.assertEqual((1,), tuplify(1))
