import requests
import urlparse
from zounds.soundfile.audio_metadata import AudioMetaData


class InternetArchive(object):
    """
    Returns prepared requests for all the audio in a specific internet archive
    collection
    """
    def __init__(self, archive_id, format_filter=None):

        self.format_filter = format_filter or \
            (lambda x: x['format'] == 'Ogg Vorbis')
        self.archive_id = archive_id

    def _get_metadata(self, data, all_files):
        if data['source'] == 'original':
            return data
        elif data['source'] == 'derivative':
            return all_files['/' + data['original']]

    def __iter__(self):
        base_url = 'https://archive.org/'
        archive_id = self.archive_id
        url = urlparse.urljoin(
            base_url, '/details/{archive_id}&output=json'.format(**locals()))
        resp = requests.get(url)

        all_files = resp.json()['files']
        for k, v in all_files.iteritems():
            if self.format_filter(v):
                sound_url = urlparse.urljoin(
                    base_url, '/download/{archive_id}{k}'.format(**locals()))
                request = requests.Request(method='GET', url=sound_url)
                yield AudioMetaData(
                    uri=request,
                    **self._get_metadata(v, all_files))
