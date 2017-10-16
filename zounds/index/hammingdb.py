import lmdb
from zounds.nputil import Growable, packed_hamming_distance
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import os
import binascii


class HammingDb(object):
    def __init__(self, path, map_size=1000000000, code_size=8):
        super(HammingDb, self).__init__()

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.env = lmdb.open(
            self.path,
            max_dbs=10,
            map_size=map_size,
            writemap=True,
            map_async=True,
            metasync=True)

        self.metadata = self.env.open_db('metadata')
        try:
            self.code_size = int(self.get_metadata('codesize'))
            if code_size and code_size != self.code_size:
                raise ValueError(
                    'Database is already initialized with code size {code_size}'
                    ', but {self.code_size} was passed to __init__'
                        .format(**locals()))
        except TypeError:
            if code_size is None:
                raise ValueError(
                    'You must supply a code size for an uninitialized database')
            if code_size % 8:
                raise ValueError('code_size must be a multiple of 8')
            self.set_metadata('codesize', str(code_size))
            self.code_size = code_size

        self.index = self.env.open_db('index')
        self._append_buffer = self._recarray(1)
        self._code_bytearray = bytearray('a' * self.code_size)
        self._code_buffer = np.frombuffer(self._code_bytearray, dtype=np.uint64)
        self._codes = None
        self._ids = set()
        self._catch_up_on_in_memory_store()

        self._thread_count = cpu_count()
        self._pool = ThreadPool(processes=self._thread_count)

    def set_metadata(self, key, value):
        with self.env.begin(write=True) as txn:
            txn.put(key, value, db=self.metadata)

    def get_metadata(self, key):
        with self.env.begin() as txn:
            return txn.get(key, db=self.metadata)

    def _catch_up_on_in_memory_store(self):
        self._initialize_in_memory_store()
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.index)
            for i, bundle in enumerate(cursor.iternext(keys=True, values=True)):
                _id, value = bundle
                if _id in self._ids:
                    continue

                code = value[:self.code_size]
                self._add_code(_id, code)

    def __len__(self):
        with self.env.begin() as txn:
            lmdb_size = txn.stat(self.index)['entries']
            if not lmdb_size:
                return 0
            return lmdb_size

    def _recarray(self, size):
        return np.recarray(
            size,
            dtype=[
                ('id', 'S32'),
                ('code', np.uint64, self.code_size // 8)],
            order='F')

    def _initialize_in_memory_store(self):
        if self._codes is not None:
            return
        self._codes = Growable(self._recarray(int(1e6)))

    def _np_code(self, code):
        self._code_bytearray[:] = code
        return self._code_buffer

    def _validate_code_size(self, code):
        code_len = len(code)
        if code_len != self.code_size:
            fmt = '''code must be equal to code_size
                    ({self.code_size}), but was {code_len}'''
            raise ValueError(fmt.format(**locals()))

    def _add_code(self, _id, code):
        arr = self._append_buffer
        arr[0]['id'] = _id
        arr[0]['code'] = self._np_code(code)
        self._codes.append(arr)
        self._ids.add(_id)

    def _check_for_external_modifications(self):
        if self.__len__() != self._codes.logical_size:
            self._catch_up_on_in_memory_store()

    def _new_id(self):
        return binascii.hexlify(os.urandom(16))

    def append(self, code, data):
        self._validate_code_size(code)
        self._initialize_in_memory_store()

        with self.env.begin(write=True) as txn:
            _id = self._new_id()
            txn.put(_id, code + data, db=self.index)
            self._add_code(_id, code)

    def _random_code(self):
        with self.env.begin() as txn:
            with txn.cursor(self.index) as cursor:
                code = None
                while not code:
                    if cursor.set_range(self._new_id()):
                        return txn.get(
                            cursor.key(), db=self.index)[:self.code_size]
                    continue

    def random_search(self, n_results, multithreaded=False):
        code = self._random_code()
        return code, self.search(code, n_results, multithreaded)

    def search(self, code, n_results, multithreaded=False):
        self._validate_code_size(code)
        self._check_for_external_modifications()
        query = self._np_code(code)

        codes = self._codes.logical_data['code']
        if codes.ndim == 1:
            codes = codes[..., None]

        if not multithreaded:
            scores = packed_hamming_distance(query, codes)
        else:
            n_codes = len(codes)
            chunksize = max(1, n_codes // self._thread_count)
            scores = np.concatenate(self._pool.map(
                lambda x: packed_hamming_distance(query, x),
                (codes[i: i + chunksize] for i in
                 xrange(0, n_codes, chunksize))))

        # indices = np.argsort(scores)[:n_results]
        indices = np.argpartition(scores, n_results)[:n_results]
        print 'DISTANCES', scores[indices]

        nearest = self._codes.logical_data[indices]['id']
        with self.env.begin() as txn:
            for _id in nearest:
                yield txn.get(_id, db=self.index)[self.code_size:]
