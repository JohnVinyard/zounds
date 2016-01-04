import json
import tornado.ioloop
import tornado.web
import httplib
from timeseries import ConstantRateTimeSeries
from PIL import Image
import numpy as np
from tornado.httputil import _parse_request_range
from flow import Decoder
from urlparse import urljoin


class ZoundsApp(object):
    def __init__(self, base_path=r'/zounds/', model=None):
        super(ZoundsApp, self).__init__()
        self.model = model
        self.base_path = base_path

    @staticmethod
    def serialize(_id, feature, slce, document):
        if feature.encoder.content_type == 'application/json':
            return \
                'application/json', \
                json.dumps(feature(_id=_id, persistence=document))

        if feature.encoder.content_type == 'audio/ogg':
            flo = feature(_id=_id, decoder=Decoder(), persistence=document)
            if slce.start:
                flo.seek(slce.start)
            if slce.stop:
                return 'audio/ogg', flo.read(slce.stop - slce.start)
            else:
                return 'audio/ogg', flo.read()

        data = feature(_id=_id, persistence=document)
        if isinstance(data, ConstantRateTimeSeries) and len(data.shape) == 2:
            data = np.abs(data)
            data *= (255. / data.max())
            new_shape = tuple(np.array(data.shape) * 10)
            img = Image.fromarray(np.rot90(data))
            img = img.resize(new_shape, resample=Image.ANTIALIAS)
            return 'image/jpg', img.convert('RGB').tobytes('jpeg', 'RGB')

    def feature_handler(self):
        document = self.model
        app = self

        class FeatureHandler(tornado.web.RequestHandler):

            def get(self, _id, feature):
                feature = getattr(document, feature)
                range_header = self.request.headers.get('Range')
                if range_header:
                    sl = slice(*_parse_request_range(range_header))
                else:
                    sl = slice(None)
                print sl
                content_type, serialized = app.serialize(
                        _id, feature, sl, document)
                self.set_header('Content-Type', content_type)
                self.set_header('Accept-Ranges', 'bytes')
                self.write(serialized[sl])
                self.set_status(httplib.OK)
                self.finish()

        return FeatureHandler

    def main_handler(self):
        document = self.model
        base_path = self.base_path
        stored_features = \
            [f.key for f in document.features.itervalues() if f.store]

        class MainHandler(tornado.web.RequestHandler):
            def get(self):
                # KLUDGE: I should figure out a way to do paging here
                _ids = list(document.database.iter_ids())
                output = dict()
                items = []
                output['items'] = items
                for _id in _ids:
                    links = [urljoin(base_path, '{_id}/{x}'.format(**locals()))
                             for x in stored_features]
                    items.append({
                        '_id': _id,
                        'links': links
                    })
                self.set_header('Content-Type', 'application/json')
                self.write(json.dumps(output))
                self.set_status(httplib.OK)
                self.finish()

        return MainHandler

    def build_app(self):
        return tornado.web.Application([
            (r'/zounds/?', self.main_handler()),
            (r'/zounds/(.+?)/(.+?)/?', self.feature_handler()),
        ])

    def start(self, port=8888):
        app = self.build_app()
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()
