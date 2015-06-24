from flow import Node
import numpy as np
import types

#TODO: Get rid of this class once I'm sure that TimeSeriesIndex works well
class Index(Node):
    
    def __init__(self, func = None, needs = None):
        super(Index, self).__init__(needs = needs)
        self._func = func
    
    def _build_recarray(self, _id, feat):
        print _id
        print feat.shape
        print feat.dtype
        assert 32 == len(_id)
        _id_dtype = 'a{l}'.format(l = len(_id)) 
        arr = np.recarray(feat.shape[0], dtype = [\
          ('code', feat.dtype, feat.shape[1:]), 
          ('_id', _id_dtype)])
        
        arr['code'][:] = feat
        arr['_id'][:] = _id
        return arr
    
    def _process(self, data):
        for _id in data.iter_ids():
            print _id
            feat = self._func(_id)
            if isinstance(feat,types.GeneratorType):
                for f in feat: 
                    yield self._build_recarray(_id, f)
            else:
                yield self._build_recarray(_id, feat)

# TODO: Make sure this can work with streaming numpy decoders as well
#class Index(Node):
#    
#    def __init__(self, func = None, needs = None):
#        super(Index, self).__init__(needs = needs)
#        self._func = func
#    
#    def _process(self, data):
#        for _id in data.iter_ids():
#            feat = self._func(_id)
#            yield (_id, feat)
#
#class TimeSeriesIndexEncoder(None):
#    
#    content_type = 'application/octet_stream'
#    
#    def __init__(self, needs = None):
#        super(TimeSeriesIndexEncoder, self).__init__(needs = needs)
#        self._init = None
#    
#    def _process(self, data):
#        pass