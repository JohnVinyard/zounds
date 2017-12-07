import requests
from zounds.soundfile import AudioMetaData
from zounds.timeseries import SR44100, AudioSamples
from predownload import PreDownload
import os
import urlparse
import numpy as np
import csv


class MusicNet(object):
    """
    Provides access to the audio and high-level metadata from MusicNet.  More
    info can be found here:
    https://homes.cs.washington.edu/~thickstn/musicnet.html
    """
    def     __init__(self, path):
        super(MusicNet, self).__init__()
        self.path = path
        self._url = \
            'https://homes.cs.washington.edu/~thickstn/media/musicnet.npz'
        self._metadata = \
            'https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv'
        self._local_data = \
            os.path.join(self.path, self._remote_filename(self._url))
        self._local_metadata = \
            os.path.join(self.path, self._remote_filename(self._metadata))
        self._samplerate = SR44100()

    def _remote_filename(self, uri):
        parsed = urlparse.urlparse(uri)
        return os.path.split(parsed.path)[-1]

    def _ensure_file_and_contents(self):
        if not os.path.exists(self._local_data):
            with open(self._local_data, 'wb') as f:
                resp = requests.get(self._url, stream=True)
                chunk_size = int(4096)
                total_bytes = int(resp.headers['Content-Length'])
                for i, chunk in enumerate(resp.iter_content(chunk_size=chunk_size)):
                    f.write(chunk)
                    progress = ((i * chunk_size) / float(total_bytes)) * 100
                    print '{progress:.2f}% complete'.format(**locals())

        if not os.path.exists(self._local_metadata):
            with open(self._local_metadata, 'wb') as f:
                resp = requests.get(self._metadata)
                f.write(resp.content)

    def __iter__(self):
        self._ensure_file_and_contents()

        metadata = dict()
        with open(self._local_metadata, 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['id']] = row

        with open(self._local_data, 'rb') as f:
            data = np.load(f)
            for k, v in data.iteritems():
                _id = k
                samples, labels = v
                samples = AudioSamples(samples, self._samplerate)
                meta = metadata[_id]
                url = \
                    'https://homes.cs.washington.edu/~thickstn/media/{_id}'\
                        .format(**locals())
                yield AudioMetaData(
                    uri=PreDownload(samples.encode().read(), url),
                    samplerate=int(self._samplerate),
                    **meta)
