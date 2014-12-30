from zounds.flow.extractor import Node
from zounds.flow.util import chunked
from requests import Session
import os
     
class ByteStream(Node):
    
    def __init__(self, chunksize = 4096, needs = None):
        super(ByteStream, self).__init__(needs = needs)
        self._chunksize = chunksize
    
    def _handle_http_request(self,data):
        s = Session()
        prepped = data.prepare()
        resp = s.send(prepped,stream = True)
        content_length = int(resp.headers['Content-Length'])
        for chunk in chunked(resp.raw, chunksize = self._chunksize):
            yield StringWithTotalLength(chunk,content_length)
    
    def _handle_local_file(self,data):
        with open(data,'rb') as f:
            content_length = int(os.path.getsize(data))
            for chunk in chunked(f, chunksize = self._chunksize):
                yield StringWithTotalLength(chunk,content_length)
    
    def _process(self,data):
        try:
            for chunk in self._handle_http_request(data):
                yield chunk
        except AttributeError:
            for chunk in self._handle_local_file(data):
                yield chunk

class StringWithTotalLength(str):
    
    def __new__(cls,s,total_length):
        o = str.__new__(cls,s)
        o.total_length = int(total_length)
        return o