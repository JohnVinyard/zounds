import requests
import time
from zounds.soundfile import AudioMetaData


class FreeSoundSearch(object):
    """
    Returns a prepared request for every result from a freesound.org search
    """

    def __init__(self, api_key, query, n_results=10, delay=0.2):
        super(FreeSoundSearch, self).__init__()
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
