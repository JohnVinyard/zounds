import simplejson
from extractor import Node,NotEnoughData

class IdentityEncoder(Node):
    
    content_type = 'application/octet-stream'
    
    def __init__(self, needs = None):
        super(IdentityEncoder,self).__init__(needs = needs)

    def _enqueue(self,data,pusher):
        self._cache = data if data else ''

class TextEncoder(IdentityEncoder):
    
    content_type = 'text/plain'
    
    def __init__(self,needs = None):
        super(TextEncoder,self).__init__(needs = needs)

class JSONEncoder(Node):
    
    content_type = 'application/json'
    
    def __init__(self, needs = None):
        super(JSONEncoder,self).__init__(needs = needs)

    def dequeue(self):
        if not self._finalized:
            raise NotEnoughData()

        return super(JSONEncoder,self)._dequeue()
        
    def _process(self,data):
        yield simplejson.dumps(data)

