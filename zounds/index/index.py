from featureflow import Node, Aggregator
import numpy as np
from bisect import bisect_left
from zounds.timeseries import TimeSlice
from zounds.nputil import packed_hamming_distance
import featureflow as ff
from zounds.timeseries import VariableRateTimeSeriesFeature
from zounds.persistence import \
    ArrayWithUnitsFeature, PackedArrayWithUnitsEncoder, ArrayWithUnitsEncoder
import json
from zounds.timeseries import ConstantRateTimeSeries
from zounds.persistence import TimeSliceEncoder, TimeSliceDecoder
import threading


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
    def __init__(self, query, offsets, indices, time_slice_builder):
        super(SearchResults, self).__init__()
        self.query = query
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
    def __init__(self, index):
        super(Scorer, self).__init__()
        self.contiguous = index.contiguous

    def decode_query(self, binary_query):
        return np.fromstring(binary_query, dtype=self.contiguous.dtype)

    def score(self, query):
        raise NotImplementedError()


class HammingDistanceScorer(Scorer):
    def __init__(self, index):
        super(HammingDistanceScorer, self).__init__(index)

    def score(self, query):
        return np.logical_xor(query, self.contiguous).sum(axis=1)


class PackedHammingDistanceScorer(Scorer):
    def __init__(self, index):
        super(PackedHammingDistanceScorer, self).__init__(index)

    def score(self, query):
        return packed_hamming_distance(
            query.view(np.uint64), self.contiguous.view(np.uint64))


class TimeSliceBuilder(object):
    def __init__(self, index):
        super(TimeSliceBuilder, self).__init__()
        self.offsets = index.offsets
        self.contiguous = index.contiguous

    def build(self, _id, diff):
        raise NotImplementedError()


class ConstantRateTimeSliceBuilder(TimeSliceBuilder):
    def __init__(self, index):
        super(ConstantRateTimeSliceBuilder, self).__init__(index)

    def build(self, _id, diff):
        time_dim = self.contiguous.dimensions[0]
        start_time = time_dim.frequency * diff
        duration = time_dim.duration
        return TimeSlice(duration, start_time)


class VariableRateTimeSliceBuilder(TimeSliceBuilder):
    def __init__(self, index, feature_func):
        super(VariableRateTimeSliceBuilder, self).__init__(index)
        self.feature_func = feature_func

    def build(self, _id, diff):
        slices = list(self.feature_func(_id))
        return slices[diff]


class Search(object):
    def __init__(self, index, scorer, time_slice_builder):
        super(Search, self).__init__()
        self.offsets = index.offsets
        self.time_slice_builder = time_slice_builder
        self.scorer = scorer

    def decode_query(self, binary_query):
        return self.scorer.decode_query(binary_query)

    def search(self, query, n_results=5):
        scores = self.scorer.score(query)
        indices = np.argsort(scores)[:n_results]
        return SearchResults(query, self.offsets, indices,
                             self.time_slice_builder)


def hamming_index(document, feature, packed=True):
    if isinstance(feature, ArrayWithUnitsFeature):
        iterator = ((d._id, feature(_id=d._id, persistence=document)) \
                    for d in document)
        constant_rate = True
        feat = ArrayWithUnitsFeature
        encoder = PackedArrayWithUnitsEncoder \
            if packed else ArrayWithUnitsEncoder
    elif isinstance(feature, VariableRateTimeSeriesFeature):
        iterator = ((d._id, feature(_id=d._id, persistence=document).slicedata) \
                    for d in document)
        constant_rate = False
        feat = ff.NumpyFeature
        encoder = ff.PackedNumpyEncoder if packed else ff.NumpyEncoder
    else:
        raise ValueError(
            'feature must be either constant or variable rate timeseries')

    class Index(ff.BaseModel):
        codes = ff.Feature(
            ff.IteratorNode,
            store=False)

        contiguous = feat(
            Contiguous,
            needs=codes,
            encoder=encoder,
            store=True)

        offsets = ff.PickleFeature(
            Offsets,
            needs=codes,
            store=True)

        def __init__(self):
            super(Index, self).__init__()
            scorer = PackedHammingDistanceScorer \
                if packed else HammingDistanceScorer
            self._scorer = scorer(self)
            if constant_rate:
                self._slice_builder = ConstantRateTimeSliceBuilder(self)
            else:
                self._slice_builder = VariableRateTimeSliceBuilder(
                    self,
                    lambda x: feature(_id=x, persistence=document).slices)
            self._search = Search(
                self,
                scorer=self._scorer,
                time_slice_builder=self._slice_builder)

        def decode_query(self, binary_query):
            return self._search.decode_query(binary_query)

        def random_query(self):
            query_index = np.random.randint(0, len(self.contiguous))
            return self.contiguous[query_index]

        def random_search(self, n_results=5):
            return self.search(self.random_query(), n_results=n_results)

        def search(self, query, n_results=5):
            return self._search.search(query, n_results=n_results)

        @classmethod
        def build(cls):
            cls.process(codes=iterator)

    return Index


class HammingIndex(object):
    # TODO: Must remember last position
    def __init__(self, event_log, hamming_db, feature, document):
        super(HammingIndex, self).__init__()
        self.document = document
        self.feature = feature
        self.hamming_db = hamming_db
        self.event_log = event_log
        self.encoder = TimeSliceEncoder()
        self.decoder = TimeSliceDecoder()
        self.thread = None

    def listen(self):
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()

    def _listen(self):
        # TODO: Must remember last position
        for timestamp, data in self.event_log.subscribe():

            # parse the data from the event stream
            data = json.loads(data)
            _id, name = data['_id'], data['name']

            # ensure that it's about the feature we're subscribed to
            if name != self.feature.key:
                continue

            # load the feature from the feature database
            feature = self.feature(_id=_id, persistence=self.document)

            try:
                arr = ConstantRateTimeSeries(feature)
            except ValueError:
                arr = feature

            # extract codes and timeslices from the feature
            for ts, data in arr.iter_slices():
                code = np.packbits(data).tostring()
                encoded_ts = dict(
                    _id=_id,
                    **self.encoder.dict(ts))
                self.hamming_db.append(code, json.dumps(encoded_ts))

    def _parse_result(self, result):
        d = json.loads(result)
        ts = TimeSlice(**self.decoder.kwargs(d))
        return d['_id'], ts

    def random_search(self, n_results, multithreaded=False):
        for result in self.hamming_db.random_search(n_results, multithreaded):
            yield self._parse_result(result)

    def search(self, feature, n_results, multithreaded=False):
        code = np.packbits(feature).tostring()
        for result in self.hamming_db.search(code, n_results, multithreaded):
            yield self._parse_result(result)
