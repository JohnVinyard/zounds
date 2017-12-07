import requests
import os
import urlparse
import tarfile
import fnmatch
import json
from predownload import PreDownload
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
        self._local_data = os.path.join(
            self.path, self._remote_filename(self._url))

    # TODO: Factor all of this out from MusicNet and NSynth
    def _remote_filename(self, uri):
        parsed = urlparse.urlparse(uri)
        return os.path.split(parsed.path)[-1]

    def _ensure_file_contents(self):
        if not os.path.exists(self._local_data):
            with open(self._local_data, 'wb') as f:
                resp = requests.get(self._url, stream=True)
                chunk_size = 4096
                total_bytes = int(resp.headers['Content-Length'])
                for i, chunk in enumerate(
                        resp.iter_content(chunk_size=chunk_size)):
                    f.write(chunk)
                    progress = ((i * chunk_size) / float(total_bytes)) * 100
                    print '{progress:.2f}% complete'.format(**locals())

    def __iter__(self):
        self._ensure_file_contents()
        json_data = None
        with tarfile.open(self._local_data) as tar:
            for info in tar:

                if fnmatch.fnmatch(info.name, '*.json'):
                    flo = tar.extractfile(member=info)
                    json_data = json.load(flo)

        with tarfile.open(self._local_data) as tar:
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
                yield AudioMetaData(uri=pdl, **json_data[_id])