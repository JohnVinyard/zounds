class Model(type):
    
    def __new__(cls,name,bases,attrs):            
        return super(Model,cls).__new__(cls,name,bases,attrs)
    
    def __init__(self,name,bases,attrs):
        super(Model,self).__init__(name,bases,attrs)
        
    @property
    def controller(self):
        raise NotImplemented()

    def __getitem__(self,_id):
        return self.controller[_id]
    
    def __setitem__(self,_id,pattern):
        self.controller[_id] = pattern
        
    def __delitem__(self,_id):
        del self.controller[_id]




