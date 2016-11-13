import unittest2
from tfrepresentation import FrequencyDimension
from frequencyscale import FrequencyBand, LinearScale, LogScale


class FrequencyDimensionTests(unittest2.TestCase):
    def test_equal(self):
        fd1 = FrequencyDimension(LinearScale(FrequencyBand(20, 10000), 100))
        fd2 = FrequencyDimension(LinearScale(FrequencyBand(20, 10000), 100))
        self.assertEqual(fd1, fd2)

    def test_not_equal(self):
        fd1 = FrequencyDimension(LinearScale(FrequencyBand(20, 10000), 100))
        fd2 = FrequencyDimension(LogScale(FrequencyBand(20, 10000), 100))
        self.assertNotEqual(fd1, fd2)
