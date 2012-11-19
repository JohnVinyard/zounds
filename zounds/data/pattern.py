from abc import ABCMeta,abstractmethod
from random import choice

from pymongo import Connection

from controller import Controller
 

class PatternController(Controller):
    
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



class MongoDbPatternController(PatternController):
    
    def __init__(self,host = 'localhost',port = None,
                 dbname = 'zounds',collection_name = 'patterns'):
        
        self.connection = Connection(*filter(lambda arg : arg,[host,port]))
        self.db = self.connection[dbname]
        self.collection = self.db[collection_name]
        self._dbname = dbname
        self._collection_name = collection_name
        # TODO: Which values should have MongoDb indexes ?
    
    def __getitem__(self,_id):
        if isinstance(_id,(str,unicode)):
            d = self.collection.find_one(_id)
            if None is d:
                raise KeyError(_id)
            return d
        
        if hasattr(_id,'__iter__'):
            # TODO: Is there a better way to fetch multiple documents by id?
            # _id is probably a set, so turn it into a list
            d = list(self.collection.find({'_id' : {'$in' : list(_id)}}))
            
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
        
        

