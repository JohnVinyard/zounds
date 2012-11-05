from abc import ABCMeta,abstractmethod
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
        
        

