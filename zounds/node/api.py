import json
import tornado.ioloop
import tornado.web
import httplib
from timeseries import ConstantRateTimeSeries
import Image
import numpy as np
from tornado.httputil import _parse_request_range
from flow import Decoder

class OtherHandler(tornado.web.RequestHandler):
    
    def get(self, a, b):
        self.write('got {a} and {b}'.format(**locals()))
        self.finish()

class ZoundsApp(object):
    
    def __init__(self, base_path = r'/zounds/', model = None):
        super(ZoundsApp, self).__init__()
        self.model = model
        self.base_path = base_path
    
    def serialize(self, _id, feature, slce):
        if feature.encoder.content_type == 'application/json':
            return 'application/json', json.dumps(feature(_id = _id))
        
        if feature.encoder.content_type == 'audio/ogg':
            flo = feature(_id = _id, decoder = Decoder())
            if slce.start: flo.seek(slce.start)
            if slce.stop: return 'audio/ogg', flo.read(slce.stop - slce.start)
            else: return 'audio/ogg', flo.read()
        
        data = feature(_id = _id)
        if isinstance(data, ConstantRateTimeSeries) and len(data.shape) == 2:
            data = np.abs(data)
            data *= (255. / data.max())
            return 'image/jpg', Image.fromarray(np.rot90(data))\
                .convert('RGB').tostring('jpeg', 'RGB')
        
    def feature_handler(self):
        Document = self.model
        app = self
        
        class FeatureHandler(tornado.web.RequestHandler):
            
            def get(self, _id, feature):
                feature = getattr(Document, feature)
                range_header = self.request.headers.get('Range')
                if range_header:
                    sl = slice(*_parse_request_range(range_header))
                else:
                    sl = slice(None)
                print sl
                content_type, serialized = app.serialize(_id, feature, sl)
                self.set_header('Content-Type', content_type)
                self.write(serialized[sl])
                self.set_status(httplib.OK)
                self.finish()
        
        return FeatureHandler
    
    def build_app(self):
        return tornado.web.Application([
            (r'/zounds/(.+?)/(.+?)/?', self.feature_handler()),
        ])
    
    def start(self, port = 8888):
        app = self.build_app()
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()