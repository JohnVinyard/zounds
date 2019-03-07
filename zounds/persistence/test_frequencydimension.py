import unittest2
from .frequencydimension import \
    FrequencyDimensionEncoder, FrequencyDimensionDecoder, \
    LinearScaleEncoderDecoder, GeometricScaleEncoderDecoder, \
    ExplicitScaleEncoderDecoder, ExplicitFrequencyDimensionEncoder, \
    ExplicitFrequencyDimensionDecoder
from zounds.spectral import \
    FrequencyDimension, FrequencyBand, LinearScale, \
    GeometricScale, ExplicitScale, FrequencyScale, ExplicitFrequencyDimension


class ScaleEncodingTests(unittest2.TestCase):
    def test_can_round_trip_linear_scale(self):
        scale = LinearScale(FrequencyBand(20, 4000), n_bands=100)
        encoder_decoder = LinearScaleEncoderDecoder()
        self.assertTrue(encoder_decoder.can_encode(scale))
        encoded = encoder_decoder.encode(scale)
        self.assertTrue(encoder_decoder.can_decode(encoded))
        decoded = encoder_decoder.decode(encoded)
        self.assertEqual(scale, decoded)

    def test_can_round_trip_geometric_scale(self):
        scale = GeometricScale(20, 5000, 0.05, n_bands=100)
        encoder_decoder = GeometricScaleEncoderDecoder()
        self.assertTrue(encoder_decoder.can_encode(scale))
        encoded = encoder_decoder.encode(scale)
        self.assertTrue(encoder_decoder.can_decode(encoded))
        decoded = encoder_decoder.decode(encoded)
        self.assertEqual(scale, decoded)

    def test_can_round_trip_explicit_scale(self):
        scale = ExplicitScale(GeometricScale(20, 5000, 0.05, n_bands=100))
        encoder_decoder = ExplicitScaleEncoderDecoder()
        self.assertTrue(encoder_decoder.can_encode(scale))
        encoded = encoder_decoder.encode(scale)
        self.assertTrue(encoder_decoder.can_decode(encoded))
        decoded = encoder_decoder.decode(encoded)
        self.assertEqual(scale, decoded)


class FrequencyDimensionTests(unittest2.TestCase):
    def setUp(self):
        self.encoder = FrequencyDimensionEncoder()
        self.decoder = FrequencyDimensionDecoder()

    def test_can_round_trip(self):
        band = FrequencyBand(20, 20000)
        scale = LinearScale(band, 50)
        dim = FrequencyDimension(scale)
        encoded = self.encoder.encode(dim)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded, FrequencyDimension)
        self.assertEqual(scale, decoded.scale)

    def test_can_round_trip_specific_scale_type(self):
        band = FrequencyBand(20, 20000)
        scale = LinearScale(band, 50)
        dim = FrequencyDimension(scale)
        encoded = self.encoder.encode(dim)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded.scale, LinearScale)
        self.assertEqual(scale, decoded.scale)

    def test_can_round_trip_geometric_scale(self):
        scale = GeometricScale(20, 5000, bandwidth_ratio=0.01, n_bands=100)
        dim = FrequencyDimension(scale)
        encoded = self.encoder.encode(dim)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded.scale, GeometricScale)
        self.assertEqual(scale, decoded.scale)

    def test_can_round_trip_explicit_scale(self):
        scale = ExplicitScale(
            GeometricScale(20, 5000, bandwidth_ratio=0.01, n_bands=100))
        dim = FrequencyDimension(scale)
        encoded = self.encoder.encode(dim)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded.scale, ExplicitScale)
        self.assertEqual(scale, decoded.scale)

    def test_raises_when_encountering_unknown_scale(self):
        band = FrequencyBand(20, 20000)
        scale = FrequencyScale(band, 50)
        dim = FrequencyDimension(scale)
        self.assertRaises(NotImplementedError, lambda: self.encoder.encode(dim))


class ExplicitFrequencyDimensionTests(unittest2.TestCase):

    def setUp(self):
        self.encoder = ExplicitFrequencyDimensionEncoder()
        self.decoder = ExplicitFrequencyDimensionDecoder()

    def test_can_round_trip(self):
        scale = GeometricScale(20, 5000, 0.05, 3)
        slices = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim = ExplicitFrequencyDimension(scale, slices)
        encoded = self.encoder.encode(dim)
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded, ExplicitFrequencyDimension)
        self.assertEqual(dim, decoded)

