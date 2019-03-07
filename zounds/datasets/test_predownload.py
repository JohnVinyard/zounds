import unittest2
from .predownload import PreDownload


class PreDownloadTest(unittest2.TestCase):
    def test_can_instantiate_predownload(self):
        pdl = PreDownload(b'something', 'https://example.com')
        self.assertEqual(b'something', pdl.read())

    def test_raises_when_no_url_is_provided(self):
        self.assertRaises(ValueError, lambda: PreDownload(b'something', None))
