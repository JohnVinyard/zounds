import unittest2
from identitydimension import IdentityDimensionEncoder, IdentityDimensionDecoder
from zounds.core import IdentityDimension, Dimension


class OtherDimension(Dimension):
    pass


class IdentityDimensionTests(unittest2.TestCase):
    def setUp(self):
        self.encoder = IdentityDimensionEncoder()
        self.decoder = IdentityDimensionDecoder()

    def test_identity_encoder_matches(self):
        self.assertTrue(self.encoder.matches(IdentityDimension()))

    def test_identity_encoder_does_not_match(self):
        self.assertFalse(self.encoder.matches(OtherDimension()))

    def test_identity_encoder_encodes(self):
        encoded = self.encoder.encode(IdentityDimension())
        self.assertIsInstance(encoded, dict)
        self.assertEqual(IdentityDimension.__name__, encoded['type'])
        self.assertEqual({}, encoded['data'])

    def test_identity_decoder_matches(self):
        encoded = {'type': IdentityDimension.__name__, 'data': {}}
        self.assertTrue(self.decoder.matches(encoded))

    def test_identity_decoder_does_not_match(self):
        encoded = {'type': OtherDimension.__name__, 'data': {}}
        self.assertFalse(self.decoder.matches(encoded))

    def test_identity_decoder_decodes(self):
        decoded = self.decoder.decode(
                {'type': IdentityDimension.__name__, 'data': {}})
        self.assertIsInstance(decoded, IdentityDimension)

    def test_roundtrip(self):
        encoded = self.encoder.encode(IdentityDimension())
        decoded = self.decoder.decode(encoded)
        self.assertIsInstance(decoded, IdentityDimension)
