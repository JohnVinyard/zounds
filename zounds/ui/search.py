import tornado
from baseapp import BaseZoundsApp, RequestContext
import base64
import urllib


class ZoundsSearch(BaseZoundsApp):
    def __init__(
            self,
            base_path=r'/zounds/',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            search=None,
            n_results=10):
        super(ZoundsSearch, self).__init__(
                base_path=base_path,
                model=model,
                visualization_feature=visualization_feature,
                audio_feature=audio_feature,
                html='search.html')
        self.n_results = n_results
        self.search = search

    def custom_routes(self):
        return [
            (r'/zounds/search', self.search_handler())
        ]

    def search_handler(self):
        app = self

        class SearchHandler(tornado.web.RequestHandler):
            def get(self):
                b64_encoded_query = urllib.unquote(
                        self.get_argument('query', default=''))
                if b64_encoded_query:
                    binary_query = base64.b64decode(b64_encoded_query)
                    query = app.search.decode_query(binary_query)
                    results = app.search.search(query, n_results=app.n_results)
                else:
                    results = app.search.random_search(n_results=app.n_results)
                context = RequestContext(value=results)
                output = app.serialize(context)
                self.set_header('Content-Type', output.content_type)
                self.write(output.data)

        return SearchHandler
