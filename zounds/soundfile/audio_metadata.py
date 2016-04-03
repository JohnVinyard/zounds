from featureflow import Node, Aggregator
import os
from soundfile import SoundFile
import requests
import json
from urlparse import urlparse


class AudioMetaData(object):
    def __init__(
            self,
            uri=None,
            samplerate=None,
            channels=None,
            licensing=None,
            description=None,
            tags=None):
        super(AudioMetaData, self).__init__()
        self.uri = uri
        self.samplerate = samplerate
        self.channels = channels
        self.licensing = licensing
        self.description = description
        self.tags = tags

    def __eq__(self, other):
        return self.uri == other.uri \
               and self.samplerate == other.samplerate \
               and self.channels == other.channels \
               and self.licensing == other.licensing \
               and self.description == other.description \
               and self.tags == other.tags


class AudioMetaDataEncoder(Aggregator, Node):
    content_type = 'application/json'

    def __init__(self, needs=None):
        super(AudioMetaDataEncoder, self).__init__(needs=needs)

    def _process(self, data):
        yield json.dumps({
            'uri': data.uri.url
            if isinstance(data.uri, requests.Request) else data.uri,
            'samplerate': data.samplerate,
            'channels': data.channels,
            'licensing': data.licensing,
            'description': data.description,
            'tags': data.tags
        })


class FreesoundOrgConfig(object):
    def __init__(self, api_key):
        super(FreesoundOrgConfig, self).__init__()
        self.api_key = api_key

    def request(self, _id):
        uri = 'http://freesound.org/apiv2/sounds/{_id}/'.format(_id=_id)
        params = {'token': self.api_key}
        metadata = requests.get(uri, params=params).json()
        request = requests.Request(
                method='GET',
                url=metadata['previews']['preview-hq-ogg'],
                params=params)
        return AudioMetaData(
                uri=request,
                samplerate=metadata['samplerate'],
                channels=metadata['channels'],
                licensing=metadata['license'],
                description=metadata['description'],
                tags=metadata['tags'])


class MetaData(Node):
    def __init__(self, needs=None):
        super(MetaData, self).__init__(needs=needs)

    @staticmethod
    def _is_url(s):
        if not isinstance(s, str):
            return False
        parsed = urlparse(s)
        return parsed.scheme and parsed.netloc

    @staticmethod
    def _is_local_file(s):
        try:
            return os.path.exists(s)
        except TypeError:
            return False

    @staticmethod
    def _is_file(s):
        try:
            s.tell()
            return True
        except AttributeError:
            return False

    def _process(self, data):
        if isinstance(data, AudioMetaData):
            yield data
        elif self._is_url(data):
            req = requests.Request(
                    method='GET',
                    url=data,
                    headers={'Range': 'bytes=0-'})
            yield AudioMetaData(uri=req)
        elif isinstance(data, requests.Request):
            if 'range' not in data.headers:
                data.headers['range'] = 'bytes=0-'
            yield AudioMetaData(uri=data)
        elif self._is_local_file(data) or self._is_file(data):
            sf = SoundFile(data)
            yield AudioMetaData(
                    uri=data,
                    samplerate=sf.samplerate,
                    channels=sf.channels)
        else:
            yield AudioMetaData(uri=data)
