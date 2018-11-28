import csv

import numpy as np

from predownload import PreDownload
from util import ensure_local_file
from zounds.soundfile import AudioMetaData
from zounds.timeseries import SR44100, AudioSamples
import os


class MusicNet(object):
    """
    Provides access to the audio and high-level metadata from MusicNet.  More
    info can be found here:
    https://homes.cs.washington.edu/~thickstn/musicnet.html

    This assumes you've downloaded and extracted the files from
    https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz to path
    """
    def __init__(self, path):
        super(MusicNet, self).__init__()
        self.path = path
        self._metadata = \
            'https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv'
        self._samplerate = SR44100()

    def __iter__(self):
        local_metadata = ensure_local_file(self._metadata, self.path)

        metadata = dict()
        with open(local_metadata, 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['id']] = row

        train_audio_path = os.path.join(self.path, 'train_data')

        for filename in os.listdir(train_audio_path):
            full_path = os.path.join(train_audio_path, filename)
            _id, ext = os.path.splitext(filename)
            url = \
                'https://homes.cs.washington.edu/~thickstn/media/{_id}'\
                    .format(**locals())
            meta = metadata[_id]
            samples = AudioSamples.from_file(full_path)
            uri = PreDownload(samples.encode().read(), url)
            yield AudioMetaData(
                uri=uri,
                samplerate=int(self._samplerate),
                **meta)
