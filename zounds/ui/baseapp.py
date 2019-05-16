import os
import re

import asyncio
import tornado.ioloop
import tornado.web
import http.client
import urllib.request, urllib.parse, urllib.error
from .contentrange import RangeRequest, UnsatisfiableRangeRequestException
from .serializer import DefaultSerializer, AudioSamplesSerializer, \
    NumpySerializer, OggVorbisSerializer, ConstantRateTimeSeriesSerializer, \
    OnsetsSerializer, SearchResultsSerializer
import threading
from uuid import uuid4


class NoMatchingSerializerException(Exception):
    pass


class RequestContext(object):
    def __init__(
            self,
            document=None,
            feature=None,
            slce=None,
            value=None):
        self.value = value
        self.slce = slce
        self.feature = feature
        self.document = document

    def __repr__(self):
        return '''RequestContext(
    document={document},
    feature={feature},
    slce={slce},
    value={value})'''.format(**self.__dict__)

    def __str__(self):
        return self.__repr__()


class BaseZoundsApp(object):
    def __init__(
            self,
            base_path=r'/zounds/',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            html=None,
            secret=None):

        super(BaseZoundsApp, self).__init__()
        self.secret = secret
        self.locals = locals
        self.globals = globals
        self.model = model
        self.visualization_feature = visualization_feature
        self.audio_feature = audio_feature
        self.base_path = base_path
        self.serializers = [
            AudioSamplesSerializer(),
            OggVorbisSerializer(),
            ConstantRateTimeSeriesSerializer(),
            DefaultSerializer('application/json'),
            DefaultSerializer('audio/ogg'),
            NumpySerializer(),
            OnsetsSerializer(
                self.visualization_feature,
                self.audio_feature,
                self.feature_path),
            SearchResultsSerializer(
                self.visualization_feature,
                self.audio_feature,
                self.feature_path)
        ]
        self._html_content = self._get_html(html)
        self.server = None
        self.thread = None

    SCRIPT_TAG = re.compile('<script\s+src="/(?P<filename>[^"]+)"></script>')

    def _get_html(self, html_filename):
        path, fn = os.path.split(__file__)
        with open(os.path.join(path, html_filename)) as f:
            html = f.read()

        to_replace = [(m.groupdict()['filename'], html[m.start(): m.end()])
                      for m in self.SCRIPT_TAG.finditer(html)]

        for filename, tag in to_replace:
            with open(os.path.join(path, filename)) as scriptfile:
                html = html.replace(
                    tag, '<script>{}</script>'.format(scriptfile.read()))

        return html

    def feature_path(self, _id, feature):
        _id = urllib.parse.quote(_id, safe='')
        return '{base_path}{_id}/{feature}'.format(
            base_path=self.base_path, _id=_id, feature=feature)

    def find_serializer(self, context):
        try:
            return next(filter(
                lambda x: x.matches(context), self.serializers))
        except StopIteration:
            raise NoMatchingSerializerException()

    def serialize(self, context):
        serializer = self.find_serializer(context)
        return serializer.serialize(context)

    def feature_handler(self):

        document = self.model
        app = self

        class FeatureHandler(tornado.web.RequestHandler):

            def get(self, _id, feature):
                doc = document(_id)
                feature = document.features[feature]
                try:
                    slce = RangeRequest(self.request.headers['Range']).range()
                except KeyError:
                    slce = slice(None)
                context = RequestContext(
                    document=doc, feature=feature, slce=slce)
                try:
                    result = app.serialize(context)
                except UnsatisfiableRangeRequestException:
                    self.set_status(http.client.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.finish()
                self.set_header('Content-Type', result.content_type)
                self.set_header('Accept-Ranges', 'bytes')
                self.write(result.data)
                self.set_header('ETag', self.compute_etag())
                self.set_status(
                    http.client.PARTIAL_CONTENT if result.is_partial
                    else http.client.OK)
                if result.content_range:
                    self.set_header('Content-Range', str(result.content_range))
                self.finish()

        return FeatureHandler

    def ui_handler(self):
        app = self

        class UIHandler(tornado.web.RequestHandler):
            def get(self):
                self.set_header('Content-Type', 'text/html')
                self.write(app._html_content)
                self.set_status(http.client.OK)
                self.finish()

        return UIHandler

    def login_handler(self):
        app = self

        class LoginHandler(tornado.web.RequestHandler):
            def get(self):
                self.write(app._get_html('login.html'))
                self.finish()

            def post(self):
                secret = self.get_body_argument('secret')
                if secret == app.secret:
                    self.set_secure_cookie('user', secret)
                    self.redirect('/')
                else:
                    self.set_status(http.client.UNAUTHORIZED)
                    self.write(app._get_html('login.html'))
                    self.finish()

        return LoginHandler

    def base_routes(self):
        return [
            (r'/', self.ui_handler()),
            (r'/zounds/(.+?)/(.+?)/?', self.feature_handler()),
            (r'/zounds/login/?', self.login_handler())
        ]

    def custom_routes(self):
        return []

    def _secure_route(self, route):

        pattern, handler_cls = route

        if 'login' in pattern:
            return pattern, handler_cls

        class Secured(handler_cls):
            def __init__(self, *args, **kwargs):
                super(Secured, self).__init__(*args, **kwargs)

            def get_current_user(self):
                return self.get_secure_cookie('user')

            @tornado.web.authenticated
            def get(self, *args, **kwargs):
                super(Secured, self).get(*args, **kwargs)

            @tornado.web.authenticated
            def post(self, *args, **kwargs):
                super(Secured, self).post(*args, **kwargs)

            @tornado.web.authenticated
            def put(self, *args, **kwargs):
                super(Secured, self).put(*args, **kwargs)

            @tornado.web.authenticated
            def delete(self, *args, **kwargs):
                super(Secured, self).delete(*args, **kwargs)

            @tornado.web.authenticated
            def head(self, *args, **kwargs):
                super(Secured, self).head(*args, **kwargs)

            @tornado.web.authenticated
            def options(self, *args, **kwargs):
                super(Secured, self).options(*args, **kwargs)

        Secured.__name__ = handler_cls.__name__
        Secured.__module__ = handler_cls.__module__

        return pattern, Secured

    def _make_app(self):
        routes = self.custom_routes() + self.base_routes()

        if self.secret:
            routes = list(map(self._secure_route, routes))

        return tornado.web.Application(
            routes,
            cookie_secret=uuid4().hex,
            login_url='/zounds/login')

    def start_in_thread(self, port=8888):
        self.thread = threading.Thread(target=self._start, args=(port,))
        self.thread.daemon = True
        self.thread.start()
        print('Interactive REPL at http://localhost:{port}'.format(port=port))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ioloop = tornado.ioloop.IOLoop.instance()
        ioloop.add_callback(ioloop.stop)
        self.server.stop()
        self.thread.join()

    def _start(self, port):
        asyncio.set_event_loop(asyncio.new_event_loop())
        app = self._make_app()
        self.server = app.listen(port)
        tornado.ioloop.IOLoop.instance().start()

    def start(self, port=8888):
        print('Interactive REPL at http://localhost:{port}'.format(**locals()))
        self._start(port)
