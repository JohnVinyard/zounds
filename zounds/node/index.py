from flow import Node, Aggregator

class Index(Node):
    
    def __init__(self, func = None, needs = None):
        super(Index, self).__init__(needs = needs)
        self._func = func
    
    def _process(self, data):
        for _id in data.iter_ids():
            print _id
            yield _id, self._func(_id)

class Contiguous(Node):
    
    def __init__(self, needs = None):
        super(Contiguous, self).__init__(needs = needs)
    
    def _process(self, data):
        _id, data = data
        yield data

class Offsets(Aggregator, Node):
    
    def __init__(self, needs = None):
        super(Offsets, self).__init__(needs = needs)
        self._cache = ([], [])
        self._offset = 0
        
    def _enqueue(self, data, pusher):
        _id, data = data
        self._cache[0].append(_id)
        self._cache[1].append(self._offset)
        self._offset += len(data)