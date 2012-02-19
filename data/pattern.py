from controller import Controller

# TODO: A big question that needs to be answered is this:
# How will I store new patterns, i.e. frames arranged in novel orders,
# so that they're searchable just like the original sounds are.
# Put another way, given a query sound, how will I search for other
# frame sequences *and* user-created patterns that are similar?
class PatternController(Controller):
    
    def __init__(self):
        Controller.__init__(self)
        
    def __getitem__(self):
        raise NotImplemented()
    
    def __setitem__(self):
        raise NotImplemented()
    
    def __delitem__(self):
        raise NotImplemented()
    
class InMemory(PatternController):
    
    def __init__(self):
        PatternController.__init__(self)
        self.store = {}
        
        

