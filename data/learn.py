from controller import Controller

class LearningController(Controller):
    '''
    An "abstract" base class for controllers that will persist and fetch
    learning pipelines
    '''
    def __init__(self):
        Controller.__init__(self)
        
    def __getitem__(self,key):
        raise NotImplemented()
    
    def __setitem__(self,key,value):
        raise NotImplemented()