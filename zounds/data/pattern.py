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
        self.store = {}
        
        

