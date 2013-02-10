from abc import ABCMeta,abstractmethod
from collections import defaultdict
from time import time

from pymongo import Connection

from controller import Controller
 

class PatternController(Controller):
    
    ADDRESS_KEY = 'address'
    ID_KEY = '_id'
    ADDRESS_ID_KEY = '.'.join([ADDRESS_KEY,ID_KEY])
    STORED_KEY = 'stored'
    SOURCE_KEY = 'source'
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        Controller.__init__(self)

    @abstractmethod        
    def __getitem__(self):
        raise NotImplemented()
    
    @abstractmethod
    def store(self,pattern):
        raise NotImplemented()
    
    @abstractmethod
    def __len__(self):
        raise NotImplemented()
    
    @abstractmethod
    def list_ids(self):
        '''
        Return a list of all pattern ids
        '''
        raise NotImplemented()

    @abstractmethod
    def contained_by(self,*frames_ids):
        '''
        Return a dictionary mapping frame ids -> lists of patterns that are
        represented by, or contained by, those frames.
        
        For example, a frames instance might contain several leaf patterns, while
        another frames instance might be the rendered version of a user-created
        pattern.
        
        KLUDGE: This concept only works with the FileSystemFrameController.Address
        implementation!!
        '''
        raise NotImplemented()

    @abstractmethod
    def recent_patterns(self,how_many = 10,exclude_user = None):
        raise NotImplemented()
    
    @abstractmethod
    def patterns_by_user(self,user_id):
        raise NotImplemented()
    
    
class InMemory(PatternController):
    
    def __init__(self):
        PatternController.__init__(self)
        self._store = {}
    
    def __getitem__(self,_id):
        try:
            # _id is a single _id. return a dictionary representing one pattern
            return self._store[_id]
        except TypeError:
            # _id is a list of _ids. return a list of dictionaries representing
            # multiple patterns
            return [self._store[i] for i in _id]
    
    def store(self,pattern):
        self._store[pattern['_id']] = pattern
    
    def __len__(self):
        return self._store.__len__()
    
    def list_ids(self):
        return set(self._store.keys())
    
    def contained_by(self,*frames_ids):
        
        if not frames_ids:
            return {}
        
        # This implementation is very inefficient, and only appropriate for
        # small, test-type scenarios
        d = defaultdict(list)
        for v in self._store.itervalues():
            if not v[self.ADDRESS_KEY]:
                continue
            
            _id = v[self.ADDRESS_KEY][self.ID_KEY]
            if _id in frames_ids: 
                d[_id].append(v)
        
        return d

    def recent_patterns(self,how_many = 10,exclude_user = None):
        raise NotImplemented()
    
    def patterns_by_user(self,user_id):
        raise NotImplemented()
            



class MongoDbPatternController(PatternController): 
    
    def __init__(self,host = 'localhost',port = None,
                 dbname = 'zounds',collection_name = 'patterns'):
        
        self.connection = Connection(*filter(lambda arg : arg,[host,port]))
        self.db = self.connection[dbname]
        self.collection = self.db[collection_name]
        self._dbname = dbname
        self._collection_name = collection_name
        
        # create an index which maps frames id -> pattern documents.
        # KLUDGE: This only works with the FileSystemFrameController.Address
        # implementation! 
        self.collection.ensure_index(self.ADDRESS_KEY)
        self.collection.ensure_index(self.STORED_KEY)
        self.collection.ensure_index(self.SOURCE_KEY)
    
    def __getitem__(self,_id):
        if isinstance(_id,(str,unicode)):
            d = self.collection.find_one(_id)
            if None is d:
                raise KeyError(_id)
            return d
        
        if hasattr(_id,'__iter__'):
            # TODO: Is there a better way to fetch multiple documents by id?
            # _id is probably a set, so turn it into a list
            d = list(self.collection.find({self.ID_KEY : {'$in' : list(_id)}}))
            
            if len(d) != len(_id):
                # some of the ids were missing
                raise KeyError(_id)
            
            return d
        
        raise ValueError('_id must be a single _id or a list of them')
    
    def store(self,pattern):
        self.collection.save(pattern)
    
    def __len__(self):
        return self.collection.count()
    
    def _cleanup(self):
        self.db.drop_collection(self._collection_name)
    
    def _distinct_ids(self):
        return self.collection.distinct('_id')
    
    def list_ids(self):
        return set(self._distinct_ids())
    
    def contained_by(self,*frames_ids):
        
        if not frames_ids:
            return {}
        
        crsr = \
            self.collection.find({self.ADDRESS_ID_KEY : {'$in' : frames_ids}})
        
        d = defaultdict(list)
        for item in crsr:
            frame_id = item['address']['_id']
            d[frame_id].append(item)
            
        return d
    
    def _recent(self,cursor,how_many):
        return cursor.sort(self.STORED_KEY,-1).limit(how_many)
    
    def recent_patterns(self,how_many = 10,exclude_user = None):
        criteria = {}
        if exclude_user:
            criteria = {self.SOURCE_KEY : {'$ne' : exclude_user}}
        cursor = self.collection.find(criteria)
        return self._recent(cursor,how_many)
    
    def patterns_by_user(self,user_id,how_many = 10):
        criteria = {self.SOURCE_KEY : user_id}
        cursor = self.collection.find(criteria)
        return self._recent(cursor, how_many)
        
        

