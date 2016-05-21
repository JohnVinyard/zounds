from featureflow import Node, Aggregator
import numpy as np
from bisect import bisect_left
from zounds.timeseries import TimeSlice
from zounds.nputil import packed_hamming_distance


class Index(Node):
    def __init__(self, func=None, needs=None):
        super(Index, self).__init__(needs=needs)
        self._func = func

    def _process(self, data):
        for _id in data.iter_ids():
            print _id
            yield _id, self._func(_id)


class Contiguous(Node):
    def __init__(self, needs=None):
        super(Contiguous, self).__init__(needs=needs)

    def _process(self, data):
        _id, data = data
        yield data


class Offsets(Aggregator, Node):
    def __init__(self, needs=None):
        super(Offsets, self).__init__(needs=needs)
        self._cache = ([], [])
        self._offset = 0

    def _enqueue(self, data, pusher):
        _id, data = data
        self._cache[0].append(_id)
        self._cache[1].append(self._offset)
        self._offset += len(data)


class SearchResults(object):
    def __init__(self, offsets, indices, time_slice_builder):
        super(SearchResults, self).__init__()
        self._indices = indices
        self._time_slice_builder = time_slice_builder
        self._offsets = offsets

    @staticmethod
    def _bisect(l, x):
        i = bisect_left(l, x)
        if i == len(l):
            return len(l) - 1
        if x < l[i]:
            i -= 1
        return i

    def __iter__(self):
        _ids, positions = self._offsets
        for i in self._indices:
            start_index = self._bisect(positions, i)
            diff = i - positions[start_index]
            _id = _ids[start_index]
            yield _id, self._time_slice_builder.build(_id, diff)


class Scorer(object):
    def __init__(self, contiguous):
        super(Scorer, self).__init__()
        self.contiguous = contiguous

    def score(self, query):
        raise NotImplementedError()


class HammingDistanceScorer(Scorer):
    def __init__(self, contiguous):
        super(HammingDistanceScorer, self).__init__(contiguous)

    def score(self, query):
        return np.logical_xor(query, self.contiguous).sum(axis=1)


class PackedHammingDistanceScorer(Scorer):
    def __init__(self, contiguous):
        super(PackedHammingDistanceScorer, self).__init__(contiguous)

    def score(self, query):
        return packed_hamming_distance(
                query.view(np.uint64), self.contiguous.view(np.uint64))


class TimeSliceBuilder(object):
    def __init__(self, contiguous, offsets):
        super(TimeSliceBuilder, self).__init__()
        self.offsets = offsets
        self.contiguous = contiguous

    def build(self, _id, diff):
        raise NotImplementedError()


class ConstantRateTimeSliceBuilder(TimeSliceBuilder):
    def __init__(self, contiguous, offsets):
        super(ConstantRateTimeSliceBuilder, self).__init__(
                contiguous, offsets)

    def build(self, _id, diff):
        start_time = self.contiguous.frequency * diff
        duration = self.contiguous.duration
        return TimeSlice(duration, start_time)


class VariableRateTimeSliceBuilder(TimeSliceBuilder):
    def __init__(self, contiguous, offsets, feature_func):
        super(VariableRateTimeSliceBuilder, self).__init__(contiguous, offsets)
        self.feature_func = feature_func

    def build(self, _id, diff):
        slices = list(self.feature_func(_id))
        return slices[diff]


class Search(object):
    def __init__(self, offsets, scorer, time_slice_builder):
        super(Search, self).__init__()
        self.offsets = offsets
        self.time_slice_builder = time_slice_builder
        self.scorer = scorer

    def search(self, query, n_results=5):
        scores = self.scorer.score(query)
        indices = np.argsort(scores)[:n_results]
        return SearchResults(self.offsets, indices, self.time_slice_builder)