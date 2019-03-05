import unittest2
from .predownload import PreDownload
from .composite import CompositeDataset
from zounds.soundfile import AudioMetaData


class DatasetA(object):
    def __init__(self):
        super(DatasetA, self).__init__()

    def __iter__(self):
        yield AudioMetaData(uri=PreDownload(b'', 'http://example.com/1'))
        yield AudioMetaData(uri=PreDownload(b'', 'http://example.com/2'))


class DatasetB(object):
    def __init__(self):
        super(DatasetB, self).__init__()

    def __iter__(self):
        yield AudioMetaData(uri=PreDownload(b'', 'http://example.com/3'))
        yield AudioMetaData(uri=PreDownload(b'', 'http://example.com/4'))


class CompositeDatasetTests(unittest2.TestCase):

    def test_composite_iterates_over_multiple_datasets(self):
        composite = CompositeDataset(DatasetA(), DatasetB())
        items = [x.uri.url for x in composite]
        self.assertEqual(4, len(items))
        self.assertIn('http://example.com/1', items)
        self.assertIn('http://example.com/2', items)
        self.assertIn('http://example.com/3', items)
        self.assertIn('http://example.com/4', items)
