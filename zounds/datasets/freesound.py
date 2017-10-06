import requests
import time
from zounds.soundfile import AudioMetaData


class FreeSoundSearch(object):
    """
    Produces an iterable of :class:`zounds.soundfile.AudioMetaData` instances
    for every result from a https://freesound.org search

    Args:
        api_key (str): Your freesound.org API key (get one here:
            (http://freesound.org/apiv2/apply/))
        query (str): The text query to perform

    Raises
        ValueError: when `api_key` and/or `query` are not supplied

    Examples:
        >>> from zounds import FreeSoundSearch
        >>> fss = FreeSoundSearch('YOUR_API_KEY', 'guitar')
        >>> iter(fss).next()
        {'description': u'Etude of Electric Guitar in Dm. Used chorus and reverberation effects. Size 6/4. Tempo 100. Gloomy and sentimental.', 'tags': [u'Etude', u'Experemental', u'Guitar', u'guitar', u'Electric', u'Chorus'], 'uri': <Request [GET]>, 'channels': 2, 'licensing': u'http://creativecommons.org/licenses/by/3.0/', 'samplerate': 44100.0}

    See Also:
        :class:`InternetArchive`
        :class:`PhatDrumLoops`
        :class:`zounds.soundfile.AudioMetaData`
    """

    def __init__(self, api_key, query, n_results=10, delay=0.2):
        super(FreeSoundSearch, self).__init__()

        if not api_key:
            raise ValueError('You must supply a freesound.org API key')

        if not query:
            raise ValueError('You must supply a text query')

        self.delay = delay
        self.n_results = n_results
        self.query = query
        self.api_key = api_key

    def _iter_results(self, link=None):
        if link:
            results = requests.get(link, params={'token': self.api_key})
        else:
            results = requests.get(
                'http://www.freesound.org/apiv2/search/text',
                params={
                    'query': self.query,
                    'token': self.api_key
                })

        results = results.json()

        for r in results['results']:
            sound_data = requests.get(
                'http://www.freesound.org/apiv2/sounds/{id}'.format(**r),
                params={'token': self.api_key}
            )
            yield sound_data.json()

            # prevent 429 "Too Many Requests" responses
            time.sleep(0.2)

        for r in self._iter_results(results['next']):
            yield r

    def __iter__(self):
        for i, data in enumerate(self._iter_results()):
            if i > self.n_results:
                break

            request = requests.Request(
                method='GET',
                url=data['previews']['preview-hq-ogg'],
                params={'token': self.api_key})

            yield AudioMetaData(
                uri=request,
                samplerate=data['samplerate'],
                channels=data['channels'],
                licensing=data['license'],
                description=data['description'],
                tags=data['tags'])
