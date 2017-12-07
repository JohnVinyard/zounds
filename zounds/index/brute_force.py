import numpy as np
from scipy.spatial.distance import cdist
from index import SearchResults
from random import choice
from zounds.timeseries import ConstantRateTimeSeries


class BruteForceSearch(object):
    def __init__(self, gen):
        index = []
        self._ids = []
        for _id, example in gen:
            index.append(example)
            crts = ConstantRateTimeSeries(example)
            for ts, _ in crts.iter_slices():
                self._ids.append((_id, ts))
        self.index = np.concatenate(index)

    def random_search(self, n_results=10):
        query = choice(self.index)
        distances = cdist(query[None, ...], self.index)
        indices = np.argsort(distances[0])[:n_results]
        return SearchResults(query, (self._ids[i] for i in indices))
