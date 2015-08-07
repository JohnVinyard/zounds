from flow import Node, Aggregator
import numpy as np
from bisect import bisect_left
from timeseries import TimeSlice
from zounds.nputil import packed_hamming_distance

class Index(Node):
    
    def __init__(self, func = None, needs = None):
        super(Index, self).__init__(needs = needs)
        self._func = func
    
    def _process(self, data):
        for _id in data.iter_ids():
            print _id
            yield _id, self._func(_id)

class Contiguous(Node):
    
    def __init__(self, needs = None):
        super(Contiguous, self).__init__(needs = needs)
    
    def _process(self, data):
        _id, data = data
        yield data

class Offsets(Aggregator, Node):
    
    def __init__(self, needs = None):
        super(Offsets, self).__init__(needs = needs)
        self._cache = ([], [])
        self._offset = 0
        
    def _enqueue(self, data, pusher):
        _id, data = data
        self._cache[0].append(_id)
        self._cache[1].append(self._offset)
        self._offset += len(data)

class Search(object):
    
    def __init__(self, contiguous, offsets):
        super(Search, self).__init__()
        self._contiguous = contiguous
        self._offsets = offsets
    
    def _score(self, query):
        raise NotImplementedError()
    
    def _bisect(self, l, x):
        i = bisect_left(l, x)
        if x < l[i]:
            i -= 1
        return i
    
    def search(self, query, n_results = 5):
        scores = self._score(query)
        indices = np.argsort(scores)[:n_results]
        for i in indices:
            start_index = self._bisect(self._offsets[1], i)
            diff = i - self._offsets[1][start_index]
            _id = self._offsets[0][start_index]
            start_time = self._contiguous.frequency * diff
            duration = self._contiguous.duration
            ts = TimeSlice(duration, start_time)
            yield _id, ts

class HammingDistanceSearch(Search):
    
    def __init__(self, contiguous, offsets):
        super(HammingDistanceSearch, self).__init__(contiguous, offsets)
    
    def _score(self, query):
        return np.logical_xor(query, self._contiguous).sum(axis = 1)

class PackedHammingDistanceSearch(Search):
    
    def __init__(self, contiguous, offsets):
        super(PackedHammingDistanceSearch, self).__init__(contiguous, offsets)
    
    def _score(self, query):
        return packed_hamming_distance(\
           query.view(np.uint64), self._contiguous.view(np.uint64))
        