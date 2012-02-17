from controller import Controller

# TODO: A big question that needs to be answered is this:
# How will I store new patterns, i.e. frames arranged in novel orders,
# so that they're searchable just like the original sounds are.
# Put another way, given a query sound, how will I search for other
# frame sequences *and* user-created patterns that are similar?
class PatternController(Controller):
    
    def __init__(self,cls):
        Controller.__init__(self,cls)
        
    def __getitem__(self):
        raise NotImplemented()
    
    def __setitem__(self):
        raise NotImplemented()
    
    def __delitem__(self):
        raise NotImplemented()
    
class InMemory(PatternController):
    
    def __init__(self,cls):
        PatternController.__init__(self,cls)
        self.store = {}
        
    def __getitem__(self,key):
        return self.cls(self.store[key])
    
    def __setitem__(self,key,pattern):
        self.store[key] = pattern._id
        
    def __delitem__(self,key):
        del self.store[key]
        

