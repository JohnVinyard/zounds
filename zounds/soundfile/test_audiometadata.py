import tempfile

import requests
import unittest2

from .audio_metadata import AudioMetaData, MetaData
from zounds.synthesize import NoiseSynthesizer
from zounds.timeseries import SR44100, Seconds


def soundfile(flo=None):
    synth = NoiseSynthesizer(SR44100())
    samples = synth.synthesize(Seconds(5)).stereo
    flo = samples.encode(flo=flo)
    return samples, flo


class AudioMetaDataTests(unittest2.TestCase):

    def test_can_handle_local_file_path(self):
        with tempfile.NamedTemporaryFile(mode='wb+') as tf:
            signal, f = soundfile(flo=tf)
            result = next(MetaData()._process(f.name))
            self.assertEqual(44100, result.samplerate)
            self.assertEqual(2, result.channels)
            self.assertIs(f.name, result.uri)

    def test_can_handle_file_like_object(self):
        samples, bio = soundfile()
        result = next(MetaData()._process(bio))
        self.assertEqual(44100, result.samplerate)
        self.assertEqual(2, result.channels)
        self.assertIs(bio, result.uri)

    def test_can_handle_url_string(self):
        url = 'http://host.com/path'
        result = next(MetaData()._process(url))
        self.assertIsInstance(result.uri, requests.Request)
        self.assertEqual(url, result.uri.url)

    def test_can_handle_request(self):
        req = requests.Request(method='GET', url='http://host.com/path')
        result = next(MetaData()._process(req))
        self.assertEqual(req, result.uri)

    def test_can_handle_audio_metadata(self):
        meta = AudioMetaData(uri='something')
        result = next(MetaData()._process(meta))
        self.assertEqual(meta, result)
