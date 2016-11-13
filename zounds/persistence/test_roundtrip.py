import unittest2
from dimension import DimensionEncoder, DimensionDecoder
from zounds.timeseries import TimeDimension, Seconds, Milliseconds
from zounds.spectral import FrequencyBand, FrequencyScale, FrequencyDimension
from zounds.core import IdentityDimension


class RoundTripTests(unittest2.TestCase):

    def roundtrip(self, o):
        encoder = DimensionEncoder()
        decoder = DimensionDecoder()
        encoded = encoder.encode(o)
        return decoder.decode(encoded)

    def test_can_round_trip_single_identity_dimension(self):
        self.fail()

    def test_can_round_trip_mixed_dimensions(self):
        self.fail()