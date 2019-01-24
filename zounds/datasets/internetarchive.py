import requests
import urlparse
from zounds.soundfile.audio_metadata import AudioMetaData
from simplejson.decoder import JSONDecodeError


class InternetArchive(object):
    """
    Produces an iterable of :class:`zounds.soundfile.AudioMetaData` instances
    for every file of a particular format from an internet archive id.

    Args:
        archive_id (str): the Internet Archive identifier
        format_filter (str): The file format to return
        attrs (dict): Extra attributes to add to the :class:`AudioMetaData`

    Raises:
        ValueError: when archive_id is not provided

    Examples:
        >>> from zounds import InternetArchive
        >>> ia = InternetArchive('Greatest_Speeches_of_the_20th_Century')
        >>> iter(ia).next()
        {'creator': u'John F. Kennedy', 'height': u'0', 'channels': None, 'genre': u'Folk', 'licensing': None, 'mtime': u'1236666800', 'samplerate': None, 'size': u'7264435', 'album': u'Great Speeches of the 20th Century [Box Set] Disc 2', 'title': u'The Cuban Missile Crisis', 'format': u'128Kbps MP3', 'source': u'original', 'description': None, 'tags': None, 'track': u'15', 'crc32': u'ace17eb5', 'md5': u'e00f4e7bd9df7bdba4db7098d1ccdfe0', 'sha1': u'e42d1f348078a11ed9a6ea9c8934a1236235c7b3', 'artist': u'John F. Kennedy', 'external-identifier': [u'urn:acoustid:ff850a0c-2efa-450f-8034-efdb31a9b696', u'urn:mb_recording_id:912cedd0-5530-4f26-972c-13d131fef06e'], 'uri': <Request [GET]>, 'length': u'454.03', 'width': u'0'}

    See Also:
        :class:`FreeSoundSearch`
        :class:`PhatDrumLoops`
        :class:`zounds.soundfile.AudioMetaData`
    """
    def __init__(self, archive_id, format_filter=None, **attrs):
        super(InternetArchive, self).__init__()

        self.attrs = attrs
        if not archive_id:
            raise ValueError('You must supply an Internet Archive id')

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

        try:
            all_files = resp.json()['files']
        except JSONDecodeError as e:
            all_files = dict()

        for k, v in all_files.iteritems():
            if self.format_filter(v):
                sound_url = urlparse.urljoin(
                    base_url, '/download/{archive_id}{k}'.format(**locals()))
                request = requests.Request(method='GET', url=sound_url)
                metadata = self._get_metadata(v, all_files)
                metadata.update(self.attrs)
                web_url = 'https://archive.org//details/{archive_id}'\
                    .format(**locals())
                metadata.update(web_url=web_url)
                yield AudioMetaData(uri=request, **metadata)
