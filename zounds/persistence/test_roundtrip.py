import unittest2
from dimension import DimensionEncoder, DimensionDecoder
from zounds.timeseries import TimeDimension, Seconds, Milliseconds
from zounds.spectral import FrequencyBand, FrequencyScale, FrequencyDimension
from zounds.core import IdentityDimension


class RoundTripTests(unittest2.TestCase):

    def roundtrip(self, o):
        encoder = DimensionEncoder()
        decoder = DimensionDecoder()
        encoded = list(encoder.encode(o))
        return list(decoder.decode(encoded))

    def test_can_round_trip_single_identity_dimension(self):
        original = [IdentityDimension()]
        restored = self.roundtrip(original)
        self.assertSequenceEqual(original, restored)

    def test_can_round_trip_mixed_dimensions(self):
        original = [
            IdentityDimension(),
            TimeDimension(Seconds(1), Milliseconds(500)),
            FrequencyDimension(FrequencyScale(FrequencyBand(100, 1000), 10))
        ]
        restored = self.roundtrip(original)
        self.assertSequenceEqual(original, restored)
