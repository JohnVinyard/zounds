from zounds.soundfile import AudioMetaData
import requests
import re
import urlparse


class PhatDrumLoops(object):
    """
    Produces an iterable of :class:`zounds.soundfile.AudioMetaData` instances
    for every drum break from http://phatdrumloops.com/beats.php

    Examples
        >>> from zounds import PhatDrumLoops
        >>> pdl = PhatDrumLoops()
        >>> iter(pdl).next()
        {'description': None, 'tags': None, 'uri': <Request [GET]>, 'channels': None, 'licensing': None, 'samplerate': None}


    See Also:
        :class:`InternetArchive`
        :class:`FreeSoundSearch`
        :class:`zounds.soundfile.AudioMetaData`
    """
    def __init__(self):
        super(PhatDrumLoops, self).__init__()

    def __iter__(self):
        resp = requests.get('http://phatdrumloops.com/beats.php')
        pattern = re.compile('href="(?P<uri>/audio/wav/[^\.]+\.wav)"')
        for m in pattern.finditer(resp.content):
            url = urlparse.urljoin('http://phatdrumloops.com',
                                   m.groupdict()['uri'])
            request = requests.Request(
                method='GET',
                url=url,
                headers={'Range': 'bytes=0-'})
            yield AudioMetaData(uri=request)
