import unittest2
from dimensions import IdentityDimension, Dimension


class IdentityDimensionTests(unittest2.TestCase):
    def test_equal(self):
        d1 = IdentityDimension()
        d2 = IdentityDimension()
        self.assertEqual(d1, d2)

    def test_not_equal(self):
        d1 = Dimension()
        d2 = IdentityDimension()
        self.assertNotEqual(d1, d2)
