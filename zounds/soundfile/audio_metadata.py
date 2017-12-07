from featureflow import Node, Aggregator
import os
from soundfile import SoundFile
import requests
import json
import io
from urlparse import urlparse
import featureflow as ff


class AudioMetaData(object):
    """
    Encapsulates metadata about a source audio file, including things like
    text descriptions and licensing information.

    Args:
        uri (requests.Request or str): uri may be either a string representing
            a network resource or a local file, or a :class:`requests.Request`
            instance
        samplerate (int): the samplerate of the source audio
        channels (int): the number of channels of the source audio
        licensing (str): The licensing agreement (if any) that applies to the
            source audio
        description (str): a text description of the source audio
        tags (str): text tags that apply to the source audio
        kwargs (dict): other arbitrary properties about the source audio

    Raises:
        ValueError: when `uri` is not provided

    See Also:
        :class:`zounds.datasets.FreeSoundSearch`
        :class:`zounds.datasets.InternetArchive`
        :class:`zounds.datasets.PhatDrumLoops`
    """
    def __init__(
            self,
            uri=None,
            samplerate=None,
            channels=None,
            licensing=None,
            description=None,
            tags=None,
            **kwargs):
        super(AudioMetaData, self).__init__()

        if not uri:
            raise ValueError('You must at least supply a uri')

        self.uri = uri
        self.samplerate = samplerate
        self.channels = channels
        self.licensing = licensing
        self.description = description
        self.tags = tags

        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @property
    def request(self):
        if hasattr(self.uri, 'url'):
            return self.uri

    def __eq__(self, other):
        return self.uri == other.uri \
               and self.samplerate == other.samplerate \
               and self.channels == other.channels \
               and self.licensing == other.licensing \
               and self.description == other.description \
               and self.tags == other.tags

    def __repr__(self):
        return self.__dict__.__str__()

    def __str__(self):
        return self.__repr__()


class AudioMetaDataEncoder(Aggregator, Node):
    content_type = 'application/json'

    def __init__(self, needs=None):
        super(AudioMetaDataEncoder, self).__init__(needs=needs)

    def _uri(self, uri):
        if isinstance(uri, requests.Request):
            return uri.url
        elif isinstance(uri, io.BytesIO) or isinstance(uri, ff.ZipWrapper):
            return None
        else:
            return uri

    def _process(self, data):
        d = dict(data.__dict__)
        d['uri'] = self._uri(data.uri)
        yield json.dumps(d)


# class FreesoundOrgConfig(object):
#     def __init__(self, api_key):
#         super(FreesoundOrgConfig, self).__init__()
#         self.api_key = api_key
#
#     def request(self, _id):
#         uri = 'http://freesound.org/apiv2/sounds/{_id}/'.format(_id=_id)
#         params = {'token': self.api_key}
#         metadata = requests.get(uri, params=params).json()
#         request = requests.Request(
#             method='GET',
#             url=metadata['previews']['preview-hq-ogg'],
#             params=params)
#         return AudioMetaData(
#             uri=request,
#             samplerate=metadata['samplerate'],
#             channels=metadata['channels'],
#             licensing=metadata['license'],
#             description=metadata['description'],
#             tags=metadata['tags'])


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
