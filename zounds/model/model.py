from zounds.environment import Environment

class Model(object):
    
    def __init__(self):
        object.__init__(self)
    
    @classmethod
    def env(cls):
        return Environment.instance
    
    @classmethod
    def controller(cls):
        return cls.env().data[cls]
    
    



