from scikits.audiolab import Sndfile
from urllib2 import urlopen
from tempfile import NamedTemporaryFile

class UrlSndFile(object):
    
    def __init__(self,url,chunksize_bytes = 60 * 96000 * 2):
        object.__init__(self)
        self.url = url
        self._chunksize_bytes = chunksize_bytes
        self._urlfile = None
        self._tmpfile = None
        self._sndfile = None
    
    def _initialize(self):
        if self._urlfile:
            return
        self._urlfile = urlopen(self.url)
        self._tmpfile = NamedTemporaryFile('w')
        self._read()
        self._sndfile = Sndfile(self._tmpfile.name,'r')
        
    def _read(self):
        self._initialize()
        data = self._urlfile.read(self._chunksize_bytes)
        self._tmpfile.write(data)
        
    def __enter__(self):
        self._initialize()
        return self
    
    def __exit__(self,t,value,tb):
        self.close()
    
    @property
    def samplerate(self):
        self._initialize()
        return self._sndfile.samplerate
    
    @property
    def channels(self):
        self._initialize()
        return self._sndfile.channels
    
    @property
    def nframes(self):
        self._initialize()
        return self._sndfile.nframes
    
    def read_frames(self,nframes):
        self._initialize()
        self._read()
        return self._sndfile.read_frames(nframes)
    
    def close(self):
        self._urlfile.close()
        self._tmpfile.close()
        self._sndfile.close()

    