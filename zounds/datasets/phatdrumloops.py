import requests
import re
import urlparse


class PhatDrumLoops(object):
    """
    Returns prepared reqeusts for every drum loop available from
    http://www.phatdrumloops.com/beats.php
    """
    def __init__(self):
        super(PhatDrumLoops, self).__init__()

    def __iter__(self):
        resp = requests.get('http://phatdrumloops.com/beats.php')
        pattern = re.compile('href="(?P<uri>/audio/wav/[^\.]+\.wav)"')
        for m in pattern.finditer(resp.content):
            url = urlparse.urljoin('http://phatdrumloops.com',
                                   m.groupdict()['uri'])
            yield requests.Request(
                method='GET',
                url=url,
                headers={'Range': 'bytes=0-'})
