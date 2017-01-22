import unittest2
from featureparser import FeatureParser
from zounds.basic import stft
from zounds.util import simple_in_memory_settings
from zounds.synthesize import NoiseSynthesizer
from zounds.timeseries import SR44100, Seconds


class FeatureParserTests(unittest2.TestCase):

    def setUp(self):
        @simple_in_memory_settings
        class Document(stft(store_fft=True)):
            pass

        synth = NoiseSynthesizer(SR44100())
        audio = synth.synthesize(Seconds(2))

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        parser = FeatureParser(Document, locals())

        self.document = Document
        self.doc = doc
        self.parser = parser

    def test_can_extract_feature(self):
        parsed_doc, feature = self.parser.parse_feature('doc.fft')
        self.assertIs(parsed_doc, self.doc)
        self.assertIs(feature, self.document.features['fft'])

    def test_can_ignore_feature_in_larger_statement(self):
        parsed_doc, feature = self.parser.parse_feature('doc.fft.shape')
        self.assertIsNone(parsed_doc)
        self.assertIsNone(feature)

    def test_can_ignore_feauture_in_expression(self):
        parsed_doc, feature = self.parser.parse_feature('doc.fft *= 10')
        self.assertIsNone(parsed_doc)
        self.assertIsNone(feature)

    def test_can_ignore_feature_in_multiply_expression(self):
        parsed_doc, feature = self.parser.parse_feature('doc.fft * 10')
        self.assertIsNone(parsed_doc)
        self.assertIsNone(feature)
