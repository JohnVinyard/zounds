import unittest2
from frequencydimension import \
    FrequencyDimensionEncoder, FrequencyDimensionDecoder
from zounds.spectral import FrequencyDimension, FrequencyScale, FrequencyBand


class FrequencyDimensionTests(unittest2.TestCase):
    def setUp(self):
        self.encoder = FrequencyDimensionEncoder()
        self.decoder = FrequencyDimensionDecoder()

    def test_can_round_trip(self):
        band = FrequencyBand(20, 20000)
        scale = FrequencyScale(band, 50)
        dim = FrequencyDimension(scale)
        encoded = self.encoder.encode(dim)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded, FrequencyDimension)
        self.assertEqual(scale, decoded.scale)
