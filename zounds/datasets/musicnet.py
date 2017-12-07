import csv

import numpy as np

from predownload import PreDownload
from util import ensure_local_file
from zounds.soundfile import AudioMetaData
from zounds.timeseries import SR44100, AudioSamples


class MusicNet(object):
    """
    Provides access to the audio and high-level metadata from MusicNet.  More
    info can be found here:
    https://homes.cs.washington.edu/~thickstn/musicnet.html
    """
    def __init__(self, path):
        super(MusicNet, self).__init__()
        self.path = path
        self._url = \
            'https://homes.cs.washington.edu/~thickstn/media/musicnet.npz'
        self._metadata = \
            'https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv'
        self._samplerate = SR44100()

    def __iter__(self):
        local_data = ensure_local_file(self._url, self.path)
        local_metadata = ensure_local_file(self._metadata, self.path)

        metadata = dict()
        with open(local_metadata, 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['id']] = row

        with open(local_data, 'rb') as f:
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
                    samplerate=int(SR44100()),
                    **meta)
