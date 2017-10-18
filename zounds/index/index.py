import json
import os
import threading
import numpy as np
from hammingdb import HammingDb
from zounds.persistence import TimeSliceEncoder, TimeSliceDecoder
from zounds.timeseries import ConstantRateTimeSeries
from zounds.timeseries import TimeSlice


class SearchResults(object):
    def __init__(self, query, results):
        super(SearchResults, self).__init__()
        self.query = query
        self.results = results

    def __iter__(self):
        for result in self.results:
            yield result


class HammingIndex(object):
    def __init__(
            self,
            document,
            feature,
            version=None,
            path='',
            db_size_bytes=1000000000,
            listen=False):

        super(HammingIndex, self).__init__()
        self.document = document
        self.feature = feature
        self.db_size_bytes = db_size_bytes
        self.path = path

        version = version or self.feature.version

        self.hamming_db_path = os.path.join(
            self.path, 'index.{self.feature.key}.{version}'
                .format(**locals()))

        try:
            self.event_log = document.event_log
        except AttributeError:
            self.event_log = None

        try:
            self.hamming_db = HammingDb(self.hamming_db_path, code_size=None)
        except ValueError:
            self.hamming_db = None

        self.encoder = TimeSliceEncoder()
        self.decoder = TimeSliceDecoder()
        self.thread = None

        if listen:
            self.listen()

    def __len__(self):
        return len(self.hamming_db)

    def stop(self):
        self.event_log.unsubscribe()

    def listen(self):
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()

    def _init_hamming_db(self, code):
        if self.hamming_db is not None:
            return
        self.hamming_db = HammingDb(self.hamming_db_path, code_size=len(code))

    def _synchronously_process_events(self):
        self._listen(raise_when_empty=True)

    def add_all(self):
        for doc in self.document:
            self.add(doc._id)

    def add(self, _id, timestamp=''):
        # load the feature from the feature database
        feature = self.feature(_id=_id, persistence=self.document)

        try:
            arr = ConstantRateTimeSeries(feature)
        except ValueError:
            arr = feature

        # extract codes and timeslices from the feature
        for ts, data in arr.iter_slices():
            code = self.encode_query(data)
            encoded_ts = dict(
                _id=_id,
                **self.encoder.dict(ts))
            self._init_hamming_db(code)
            self.hamming_db.append(code, json.dumps(encoded_ts))
            self.hamming_db.set_metadata('timestamp', bytes(timestamp))

    def _listen(self, raise_when_empty=False):

        if self.hamming_db is not None:
            last_timestamp = self.hamming_db.get_metadata('timestamp') or ''
        else:
            last_timestamp = ''

        if not self.event_log:
            raise ValueError(
                '{self.document} must have an event log configured'
                    .format(**locals()))

        subscription = self.event_log.subscribe(
            last_id=last_timestamp, raise_when_empty=raise_when_empty)

        for timestamp, data in subscription:

            # parse the data from the event stream
            data = json.loads(data)
            _id, name, version = data['_id'], data['name'], data['version']

            # ensure that it's about the feature we're subscribed to
            if name != self.feature.key or version != self.feature.version:
                continue

            self.add(_id, timestamp)

    def _parse_result(self, result):
        d = json.loads(result)
        ts = TimeSlice(**self.decoder.kwargs(d))
        return d['_id'], ts

    def decode_query(self, binary_query):
        packed = np.fromstring(binary_query, dtype=np.uint8)
        return np.unpackbits(packed)

    def encode_query(self, feature):
        return np.packbits(feature).tostring()

    def random_search(self, n_results, multithreaded=False):
        code, raw_results = self.hamming_db.random_search(n_results,
                                                          multithreaded)
        parsed_results = (self._parse_result(r) for r in raw_results)
        return SearchResults(code, parsed_results)

    def search(self, feature, n_results, multithreaded=False):
        code = self.encode_query(feature)
        raw_results = self.hamming_db.search(code, n_results, multithreaded)
        parsed_results = (self._parse_result(r) for r in raw_results)
        return SearchResults(code, parsed_results)
