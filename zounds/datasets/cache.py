import os
import hashlib
import requests


class DataSetCache(object):
    def __init__(self, path, dataset):
        super(DataSetCache, self).__init__()
        self.dataset = dataset
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

    def __iter__(self):
        for request in self.dataset:
            h = hashlib.md5(request.url).hexdigest()
            p = os.path.join(self.path, h)

            if not os.path.exists(p):
                s = requests.Session()
                prepped = request.prepare()
                resp = s.send(prepped, stream=True)

                resp.raise_for_status()

                with open(p, 'wb') as f:
                    for chunk in resp.iter_content(1024):
                        f.write(chunk)

            yield p
