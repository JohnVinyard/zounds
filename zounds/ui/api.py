import sys
import json
import tornado.ioloop
import tornado.web
import httplib
import traceback
from cStringIO import StringIO
import uuid
from baseapp import BaseZoundsApp, NoMatchingSerializerException, RequestContext
from featureparser import FeatureParser


class ZoundsApp(BaseZoundsApp):
    """
    Adds an in-browser REPL to the base zounds application
    """
    def __init__(
            self,
            base_path=r'/zounds/',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            globals={},
            locals={},
            html='index.html',
            secret=None):

        super(ZoundsApp, self).__init__(
            base_path=base_path,
            model=model,
            visualization_feature=visualization_feature,
            audio_feature=audio_feature,
            html=html,
            secret=secret)

        self.globals = globals
        self.locals = locals
        self.temp = {}

    def custom_routes(self):
        return [
            (r'/zounds/temp/(.+?)/?', self.temp_handler()),
            (r'/zounds/repl/?', self.repl_handler())
        ]

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

            def _add_url(self, statement, output, value):
                parser = FeatureParser(document, locals)
                doc, feature = parser.parse_feature(statement)
                try:
                    context = RequestContext(
                            document=doc,
                            feature=feature,
                            value=value,
                            slce=slice(None))
                    result = app.serialize(context)
                    temp_id = uuid.uuid4().hex
                    app.temp[temp_id] = result
                    output['url'] = '/zounds/temp/{temp_id}'.format(
                            temp_id=temp_id)
                    output['contentType'] = result.content_type
                except NoMatchingSerializerException:
                    pass

            def post(self):
                statement = self.request.body
                self.set_header('Content-Type', 'application/json')
                output = dict()

                try:
                    orig_stdout = sys.stdout
                    sys.stdout = sio = StringIO()
                    try:
                        value = eval(statement, globals, locals)
                        output['result'] = str(value)
                        self._add_url(statement, output, value)
                    except SyntaxError:
                        exec (statement, globals, locals)
                        sio.seek(0)
                        output['result'] = sio.read()
                    self.set_status(httplib.OK)
                except:
                    output['error'] = traceback.format_exc()
                    self.set_status(httplib.BAD_REQUEST)
                finally:
                    sys.stdout = orig_stdout

                self.write(json.dumps(output))
                self.finish()

        return ReplHandler
