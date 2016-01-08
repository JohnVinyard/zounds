import os
import json
import tornado.ioloop
import tornado.web
import httplib
from timeseries import ConstantRateTimeSeriesFeature
import numpy as np
from tornado.httputil import _parse_request_range
from flow import Decoder
from urlparse import urljoin
import traceback
from matplotlib import pyplot as plt
from io import BytesIO
import ast


class NoMatchingSerializerException(Exception):
    pass


class DefaultSerializer(object):

    def __init__(self, content_type):
        super(DefaultSerializer, self).__init__()
        self._content_type = content_type

    def matches(self, feature):
        return feature.encoder.content_type == self._content_type

    @property
    def content_type(self):
        return self._content_type

    def serialize(self, _id, feature, slce, document):
        flo = feature(_id=_id, decoder=Decoder(), persistence=document)
        if slce.start:
            flo.seek(slce.start)
        if slce.stop:
            value = flo.read(slce.stop - slce.start)
        else:
            value = flo.read()
        return self._content_type, value


class ConstantRateTimeSeriesSerializer(object):

    def __init__(self):
        super(ConstantRateTimeSeriesSerializer, self).__init__()

    def matches(self, feature):
        return isinstance(feature, ConstantRateTimeSeriesFeature)

    @property
    def content_type(self):
        return 'image/png'

    def serialize(self, _id, feature, slce, document):
        data = feature(_id=_id, persistence=document)
        plt.figure()
        if len(data.shape) == 1:
            plt.plot(data)
        elif len(data.shape) == 2:
            plt.matshow(data.T)
        else:
            raise ValueError('cannot handle dimensions > 2')
        bio = BytesIO()
        plt.savefig(bio, format='png')
        bio.seek(0)
        return self.content_type, bio.read()


class ZoundsApp(object):

    def __init__(
            self, base_path=r'/zounds/', model=None, globals={}, locals={}):

        super(ZoundsApp, self).__init__()
        self.locals = locals
        self.globals = globals
        self.model = model
        self.base_path = base_path
        self.serializers = [
            DefaultSerializer('application/json'),
            DefaultSerializer('audio/ogg'),
            ConstantRateTimeSeriesSerializer()
        ]

        path, fn = os.path.split(__file__)
        with open(os.path.join(path, 'zounds.js')) as f:
            script = f.read()

        with open(os.path.join(path, 'index.html')) as f:
            self._html_content = f.read().replace(
                '<script src="/zounds.js"></script>',
                '<script>{}</script>'.format(script))

    def feature_link(self, _id, feature_name):
        return urljoin(
            self.base_path,
            '{_id}/{feature}'.format(_id=_id, feature=feature_name))

    def find_serializer(self, feature):
        try:
            return filter(
                lambda x: x.matches(feature), self.serializers)[0]
        except IndexError:
            raise NoMatchingSerializerException()

    def serialize(self, _id, feature, slce, document):
        return self\
            .find_serializer(feature)\
            .serialize(_id, feature, slce, document)

    def content_type(self, feature):
        return self.find_serializer(feature).content_type

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
                content_type, serialized = app.serialize(
                        _id, feature, sl, document)
                self.set_header('Content-Type', content_type)
                self.set_header('Accept-Ranges', 'bytes')
                self.write(serialized[sl])
                self.set_status(httplib.OK)
                self.finish()

        return FeatureHandler

    def repl_handler(self):
        document = self.model
        globals = self.globals
        locals = self.locals
        app = self

        class ReplHandler(tornado.web.RequestHandler):

            def _add_url(self, statement, output):
                root = ast.parse(statement)
                nodes = list(ast.walk(root))
                doc = None
                feature_name = None

                for node in nodes:
                    if doc and feature_name:
                        break

                    if isinstance(node, ast.Name) \
                            and node.id in locals \
                            and isinstance(locals[node.id], document):
                        doc = locals[node.id]
                        continue

                    if isinstance(node, ast.Attribute) \
                            and node.attr in document.features:
                        feature_name = node.attr
                        continue

                if not (doc and feature_name):
                    return

                feature = document.features[feature_name]
                output['url'] = app.feature_link(doc._id, feature_name)
                output['contentType'] = app.content_type(feature)

            def post(self):
                statement = self.request.body
                self.set_header('Content-Type', 'application/json')
                output = dict()
                try:
                    try:
                        value = eval(statement, globals, locals)
                        output['result'] = str(value)
                    except SyntaxError:
                        exec(statement, globals, locals)
                        output['result'] = ''
                    self.set_status(httplib.OK)
                except:
                    output['error'] = traceback.format_exc()
                    self.set_status(httplib.BAD_REQUEST)
                self._add_url(statement, output)
                self.write(json.dumps(output))
                self.finish()

        return ReplHandler

    def main_handler(self):
        document = self.model
        stored_features = \
            [f.key for f in document.features.itervalues() if f.store]
        app = self

        class MainHandler(tornado.web.RequestHandler):

            def get(self):
                # KLUDGE: I should figure out a way to do paging here
                _ids = list(document.database.iter_ids())
                output = dict()
                items = []
                output['items'] = items
                for _id in _ids:
                    links = [app.feature_link(_id, x) for x in stored_features]
                    items.append({
                        '_id': _id,
                        'links': links
                    })
                self.set_header('Content-Type', 'application/json')
                self.write(json.dumps(output))
                self.set_status(httplib.OK)
                self.finish()

        return MainHandler

    def ui_handler(self):
        app = self

        class UIHandler(tornado.web.RequestHandler):

            def get(self):
                self.set_header('Content-Type', 'text-html')
                self.write(app._html_content)
                self.set_status(httplib.OK)
                self.finish()

        return UIHandler

    def build_app(self):
        return tornado.web.Application([
            (r'/', self.ui_handler()),
            (r'/zounds/?', self.main_handler()),
            (r'/zounds/(.+?)/(.+?)/?', self.feature_handler()),
            (r'/zounds/repl/?', self.repl_handler())
        ])

    def start(self, port=8888):
        print __file__
        app = self.build_app()
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()

