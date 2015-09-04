import lmdb
from flow import Database, KeyBuilder, dependency
from io import BytesIO

class WriteStream(object):
    
    def __init__(self, _id, env, db):
        self._id = _id
        self.db = db
        self.env = env
        self.buf = BytesIO()

    def __enter__(self):
        return self

    def __exit__(self, t, value, traceback):
        self.buf.seek(0)
        with self.env.begin(write = True) as txn:
            txn.put(self._id, self.buf.read(), db = self.db)
    
    def write(self, data):
        self.buf.write(data)

# TODO: Should I do a sub-database per feature, or organize the keys as 
# feature:id?  Does it matter?  Are these equivalent?
class LmdbDatabase(Database):
    '''
    '''
    
    def __init__(self, path):
        super(Database, self).__init__()
        self.path = path
        self.env = lmdb.open(self.path, max_dbs = 10)
        self.dbs = dict()
    
    def _get_db(self, key):
        _id, feature = self.key_builder.decompose(key)
        try:
            return _id, self.dbs[feature]
        except KeyError:
            db = self.env.open_db(feature)
            self.dbs[feature] = db
            return _id, db
    
    @property
    @dependency(KeyBuilder)
    def key_builder(self): pass
    
    def write_stream(self, key, content_type):
        # Create a sub-database for the feature if it doesn't already exist,
        # and return a file like object that writes the data to lmdb once
        # all data has been written and __exit__ has been called
        _id, db = self._get_db(key)
        return WriteStream(_id, self.env, db)
    
    def read_stream(self, key):
        # Can I use a buffer outside of a transaction, or do I need to return
        # a file like object here that manages the transaction in its
        # __enter__ and __exit__ methods?
        _id, db = self._get_db(key)
        with self.env.begin(buffers = True) as txn:
            buf = txn.get(_id, db = db)
        if buf is None:
            raise KeyError(key)
        return BytesIO(buf)
    
    def iter_ids(self):
        # iterate over all the keys in a single sub-database
        db = self.dbs.values()[0]
        with self.env.begin() as txn:
            cursor = txn.cursor(db)
            for _id in cursor.iternext(keys = True, values = False):
                yield _id

    def __contains__(self, key):
        _id, db = self._get_db(key)
        with self.env.begin(buffers = True) as txn:
            buf = txn.get(_id, db = db)
        return buf is not None