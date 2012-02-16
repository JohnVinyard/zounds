from controller import Controller

class PatternController(Controller):
    
    def __init__(self,cls):
        Controller.__init__(self,cls)
        
    def __getitem__(self):
        raise NotImplemented()
    
    def __setitem__(self):
        raise NotImplemented()
    
class InMemory(PatternController):
    
    def __init__(self,cls):
        PatternController.__init__(self,cls)
        self.store = {}
        
    def __getitem__(self,key):
        return self.cls(self.store[key])
    
    def __setitem__(self,key,pattern):
        self.store[key] = pattern._id
        

