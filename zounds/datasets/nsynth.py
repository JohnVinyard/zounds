import fnmatch
import json
import os
import tarfile

from .predownload import PreDownload
from .util import ensure_local_file
from zounds.soundfile import AudioMetaData


class NSynth(object):
    """
    Provides acess to the NSynth dataset:
    https://magenta.tensorflow.org/datasets/nsynth

    Currently only downloads and iterates over the validation set
    """

    def __init__(self, path):
        super(NSynth, self).__init__()
        self.path = path
        self._url = \
            'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz'

    def __iter__(self):
        local_data = ensure_local_file(self._url, self.path)

        json_data = None
        with tarfile.open(local_data) as tar:
            for info in tar:

                if fnmatch.fnmatch(info.name, '*.json'):
                    flo = tar.extractfile(member=info)
                    json_data = json.load(flo)

        with tarfile.open(local_data) as tar:
            for info in tar:
                if fnmatch.fnmatch(info.name, '*.json'):
                    continue
                if not info.isfile():
                    continue
                path_segments = os.path.split(info.name)
                _id = os.path.splitext(path_segments[1])[0]
                wav_flo = tar.extractfile(member=info)
                url = \
                    'https://magenta.tensorflow.org/datasets/nsynth/{_id}' \
                        .format(**locals())
                pdl = PreDownload(wav_flo.read(), url)
                yield AudioMetaData(
                    uri=pdl,
                    web_url='https://magenta.tensorflow.org/datasets/nsynth',
                    **json_data[_id])
