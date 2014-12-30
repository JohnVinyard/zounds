from StringIO import StringIO
from uuid import uuid4

from extractor import Node
from dependency_injection import dependency

class IdProvider(object):
    '''
    Marker class for object that returns new ids
    '''
    def new_id(self):
        raise NotImplemented()

class UuidProvider(IdProvider):

    def __init__(self):
        super(UuidProvider,self).__init__()

    def new_id(self):
        return uuid4().hex

class KeyBuilder(object):
    '''
    Marker class for an algorithm to build keys
    from "document" id and feature name
    '''
    def build(self,_id,feature_name):
        raise NotImplemented()

class StringDelimitedKeyBuilder(KeyBuilder):

    def __init__(self,seperator = ':'):
        super(StringDelimitedKeyBuilder,self).__init__()
        self.seperator = seperator

    def build(self,_id,feature_name):
        return '{_id}{sep}{feature}'.format(\
            _id = _id,
            sep = self.seperator,
            feature = feature_name)

class Database(object):
    '''
    Marker class for a datastore
    '''
    # TODO: Maybe this should just be open(), since it returns a file-like 
    # object
    def write_stream(self,key,content_type):
        raise NotImplemented()
    
    # TODO: Maybe this should just be open(), since it returns a file-like
    # object
    def read_stream(self,key):
        raise NotImplemented()

class InMemoryDatabase(Database):

    def __init__(self):
        super(InMemoryDatabase,self).__init__()
        self._dict = dict()

    def write_stream(self,key,content_type):
        sio = StringIO()
        self._dict[key] = sio
        def hijacked_close():
            sio.seek(0)
            self._dict[key] = sio.read()
            sio._old_close()
        sio._old_close = sio.close
        sio.close = hijacked_close
        return sio

    def read_stream(self,key):
        return StringIO(self._dict[key])

class DataWriter(Node):
    
    def __init__(\
            self, 
            needs = None,
            _id = None,  
            feature_name = None):
        
        super(DataWriter,self).__init__(needs = needs)
        self._id = _id
        self.feature_name = feature_name
        self.content_type = needs.content_type
        self._stream = None

    @property
    @dependency(KeyBuilder)
    def key_builder(self): pass
    
    @property
    def key(self):
        return self.key_builder.build(self._id,self.feature_name)
    
    @property
    @dependency(Database)
    def db(self): pass

    def __enter__(self):
        self._stream = self.db.write_stream(self.key,self.content_type)
        return self

    def __exit__(self,t,value,traceback):
        self._stream.close()
        
    def _process(self,data):
        yield self._stream.write(data)

class StringIODataWriter(Node):

    def __init__(self,needs = None,_id = None,feature_name = None):
        super(StringIODataWriter,self).__init__(needs = needs)
        self._id = _id
        self.feature_name = feature_name
        self.content_type = needs.content_type
        self._stream = StringIO()

    def _process(self,data):
        yield self._stream.write(data)

class DataReader(object):
    '''
    Marker class for object that reads from the datastore
    '''
    def __init__(self,_id = None, feature = None):
        super(DataReader,self).__init__()

class DataReaderFactory(object):
    
    @property
    @dependency(Database)
    def database(self):
        pass

    @property
    @dependency(KeyBuilder)
    def key_builder(self): pass
    
    def __call__(self,_id,feature_name):
        return self.database.read_stream(\
            self.key_builder.build(_id,feature_name))

