import requests
import urlparse


class InternetArchive(object):
    """
    Returns prepared requests for all the audio in a specific internet archive
    collection
    """
    def __init__(self, archive_id, format_filter=None):

        self.format_filter = format_filter or \
            (lambda x: x['format'] == 'Ogg Vorbis')
        self.archive_id = archive_id

    def __iter__(self):
        base_url = 'https://archive.org/'
        archive_id = self.archive_id
        url = urlparse.urljoin(
            base_url, '/details/{archive_id}&output=json'.format(**locals()))
        resp = requests.get(url)
        for k, v in resp.json()['files'].iteritems():
            if self.format_filter(v):
                sound_url = urlparse.urljoin(
                    base_url, '/download/{archive_id}{k}'.format(**locals()))
                yield requests.Request(method='GET', url=sound_url)
