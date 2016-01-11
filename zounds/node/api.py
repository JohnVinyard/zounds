import os
import json
import tornado.ioloop
import tornado.web
import httplib
import numpy as np
from flow import Decoder
from urlparse import urljoin
import traceback
from matplotlib import pyplot as plt
from io import BytesIO
import ast
import uuid
import datetime


class NoMatchingSerializerException(Exception):
    pass


class TempResult(object):

    def __init__(self, data, content_type):
        self.data = data
        self.content_type = content_type
        self.timestamp = datetime.datetime.utcnow()


class DefaultSerializer(object):

    def __init__(self, content_type):
        super(DefaultSerializer, self).__init__()
        self._content_type = content_type

    def matches(self, feature, value):
        if feature is None:
            return False
        return feature.encoder.content_type == self._content_type

    @property
    def content_type(self):
        return self._content_type

    def serialize(self, feature, slce, document, value):
        flo = feature(_id=document._id, decoder=Decoder(), persistence=document)
        if slce.start:
            flo.seek(slce.start)
        if slce.stop:
            value = flo.read(slce.stop - slce.start)
        else:
            value = flo.read()
        return TempResult(value, self.content_type)


class NumpySerializer(object):

    def __init__(self):
        super(NumpySerializer, self).__init__()

    def matches(self, feature, value):
        return isinstance(value, np.ndarray) and len(value.shape) < 3

    @property
    def content_type(self):
        return 'image/png'

    # TODO: Bundle all these parameters up in some kind of context object
    def serialize(self, feature, slce, document, value):
        data = value
        fig = plt.figure()
        if len(data.shape) == 1:
            plt.plot(data)
        elif len(data.shape) == 2:
            mat = plt.matshow(np.rot90(data), cmap=plt.cm.gray)
            mat.axes.get_xaxis().set_visible(False)
            mat.axes.get_yaxis().set_visible(False)
        else:
            raise ValueError('cannot handle dimensions > 2')
        bio = BytesIO()
        plt.savefig(bio, bbox_inches='tight', pad_inches=0, format='png')
        bio.seek(0)
        fig.clf()
        plt.close()
        return TempResult(bio.read(), self.content_type)


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
            NumpySerializer()
        ]
        self.temp = {}

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

    def find_serializer(self, feature, value):
        try:
            return filter(
                lambda x: x.matches(feature, value), self.serializers)[0]
        except IndexError:
            raise NoMatchingSerializerException()

    def serialize(self, feature, slce, document, value):
        return self\
            .find_serializer(feature, value)\
            .serialize(feature, slce, document, value)

    def temp_handler(self):

        app = self

        class TempHandler(tornado.web.RequestHandler):

            def get(self, _id):
                try:
                    result = app.temp[_id]
                except KeyError:
                    self.set_status(httplib.NOT_FOUND)
                    self.finish()
                    return
                self.set_header('Content-Type', result.content_type)
                self.set_header('Accept-Ranges', 'bytes')
                self.write(result.data)
                self.set_status(httplib.OK)
                self.finish()

        return TempHandler

    def repl_handler(self):
        document = self.model
        globals = self.globals
        locals = self.locals
        app = self

        class ReplHandler(tornado.web.RequestHandler):

            def _extract_feature(self, statement):
                root = ast.parse(statement)
                nodes = list(ast.walk(root))
                doc = None
                feature = None
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

                if feature_name:
                    feature = document.features[feature_name]

                return doc, feature

            def _add_url(self, statement, output, value):
                doc, feature = self._extract_feature(statement)
                try:
                    result = app.serialize(feature, slice(None), doc, value)
                    temp_id = uuid.uuid4().hex
                    app.temp[temp_id] = result
                    output['url'] = '/zounds/temp/{temp_id}'.format(temp_id=temp_id)
                    output['contentType'] = result.content_type
                except NoMatchingSerializerException:
                    pass

            def post(self):
                statement = self.request.body
                self.set_header('Content-Type', 'application/json')
                output = dict()

                try:
                    try:
                        value = eval(statement, globals, locals)
                        output['result'] = str(value)
                        self._add_url(statement, output, value)
                    except SyntaxError:
                        exec(statement, globals, locals)
                        output['result'] = ''
                    self.set_status(httplib.OK)
                except:
                    output['error'] = traceback.format_exc()
                    self.set_status(httplib.BAD_REQUEST)

                self.write(json.dumps(output))
                self.finish()

        return ReplHandler

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
            (r'/zounds/temp/(.+?)/?', self.temp_handler()),
            (r'/zounds/repl/?', self.repl_handler())
        ])

    def start(self, port=8888):
        app = self.build_app()
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()

