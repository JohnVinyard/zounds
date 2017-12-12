import numpy as np
from scipy.spatial.distance import cdist
from index import SearchResults
from random import choice
from zounds.timeseries import ConstantRateTimeSeries
from zounds.nputil import packed_hamming_distance


class BaseBruteForceSearch(object):
    def __init__(self, gen):
        index = []
        self._ids = []
        for _id, example in gen:
            index.append(example)
            crts = ConstantRateTimeSeries(example)
            for ts, _ in crts.iter_slices():
                self._ids.append((_id, ts))
        self.index = np.concatenate(index)

    def search(self, query, n_results=10):
        raise NotImplementedError()

    def random_search(self, n_results=10):
        query = choice(self.index)
        return self.search(query, n_results)


# class BruteForceSearch(object):
#     def __init__(self, gen):
#         index = []
#         self._ids = []
#         for _id, example in gen:
#             index.append(example)
#             crts = ConstantRateTimeSeries(example)
#             for ts, _ in crts.iter_slices():
#                 self._ids.append((_id, ts))
#         self.index = np.concatenate(index)
#
#     def random_search(self, n_results=10):
#         query = choice(self.index)
#         distances = cdist(query[None, ...], self.index)
#         indices = np.argsort(distances[0])[:n_results]
#         return SearchResults(query, (self._ids[i] for i in indices))

class BruteForceSearch(BaseBruteForceSearch):
    def __init__(self, gen):
        super(BruteForceSearch, self).__init__(gen)

    def search(self, query, n_results=10):
        distances = cdist(query[None, ...], self.index)
        indices = np.argsort(distances[0])[:n_results]
        return SearchResults(query, (self._ids[i] for i in indices))


class HammingDistanceBruteForceSearch(BaseBruteForceSearch):
    def __init__(self, gen):
        super(HammingDistanceBruteForceSearch, self).__init__(gen)

    def search(self, query, n_results=10):
        scores = packed_hamming_distance(query, self.index)
        indices = np.argsort(scores)[:n_results]
        return SearchResults(query, (self._ids[i] for i in indices))
