import unittest2
from audio_metadata import AudioMetaData, MetaData
import numpy as np
from io import BytesIO
from soundfile import SoundFile
import tempfile
import requests


def soundfile(hz=440, seconds=5., sr=44100., flo=None):
    bio = flo or BytesIO()
    s = np.random.random_sample((int(seconds * sr), 2))
    with SoundFile(
            bio,
            mode='w',
            channels=2,
            format='WAV',
            subtype='PCM_16',
            samplerate=int(sr)) as f:
        f.write(s)
    bio.seek(0)
    return s, bio


class AudioMetaDataTests(unittest2.TestCase):

    def test_can_handle_local_file_path(self):
        with tempfile.NamedTemporaryFile(mode='w+') as tf:
            signal, f = soundfile(flo=tf)
            result = MetaData()._process(f.name).next()
            self.assertEqual(44100, result.samplerate)
            self.assertEqual(2, result.channels)
            self.assertIs(f.name, result.uri)

    def test_can_handle_file_like_object(self):
        samples, bio = soundfile()
        result = MetaData()._process(bio).next()
        self.assertEqual(44100, result.samplerate)
        self.assertEqual(2, result.channels)
        self.assertIs(bio, result.uri)

    def test_can_handle_url_string(self):
        url = 'http://host.com/path'
        result = MetaData()._process(url).next()
        self.assertIsInstance(result.uri, requests.Request)
        self.assertEqual(url, result.uri.url)

    def test_can_handle_request(self):
        req = requests.Request(method='GET', url='http://host.com/path')
        result = MetaData()._process(req).next()
        self.assertEqual(req, result.uri)

    def test_can_handle_audio_metadata(self):
        meta = AudioMetaData(uri='something')
        result = MetaData()._process(meta).next()
        self.assertEqual(meta, result)
