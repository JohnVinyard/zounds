from abc import ABCMeta,abstractmethod
from controller import Controller
 


class PatternController(Controller):
    
    def __init__(self):
        Controller.__init__(self)

    @abstractmethod        
    def __getitem__(self):
        raise NotImplemented()
    
    
    
class InMemory(PatternController):
    
    def __init__(self):
        PatternController.__init__(self)
        self._store = {}
    
    def __getitem__(self,_id):
        try:
            return self._store[_id]
        except TypeError:
            # _id is a list of _ids
            return dict((i,self._store[_id]) for i in _id)
    
    def store(self,pattern):
        self._store[pattern._id] = pattern
        
        

