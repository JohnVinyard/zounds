from zounds.flow.extractor import Node
from pysoundfile import SoundFile
from io import BytesIO

class AudioStream(Node):
    
    def __init__(self, sum_to_mono = True, needs = None):
        super(AudioStream, self).__init__(needs = needs)
        self._sum_to_mono = sum_to_mono
        self._buf = None
        self._sf = None
    
    def _get_samples(self):
        samples = self._sf.read(44100 * 20)
        if self._sum_to_mono:
            samples = samples.sum(axis = 1) * 0.5
        return samples
    
    def _process(self,data):
        b = data
        content_length = b.total_length
        if self._buf is None:
            self._buf = MemoryBuffer(content_length)
        
        self._buf.write(b)
        
        if self._sf is None:
            self._sf = SoundFile(self._buf,virtual_io = True)
        samples = self._get_samples()
        while samples.size:
            yield samples
            samples = self._get_samples()

class MemoryBuffer(object):
    
    def __init__(self,content_length, max_size = 10 * 1024 * 1024):
        self._content_length = content_length
        self._buf = BytesIO()
        self._max_size = max_size
    
    def __len__(self):
        return self._content_length
    
    def read(self,count): 
        if count == -1:
            return self._buf.read()
        return self._buf.read(count)
    
    def readinto(self,buf):
        data = self.read(len(buf))
        ld = len(data)
        buf[:ld] = data
        return ld
    
    def write(self,data):
        read_pos = self._buf.tell()
        if read_pos > self._max_size:
            new_buf = BytesIO()
            new_buf.write(self._buf.read())
            self._buf = new_buf
            read_pos = 0
        self._buf.seek(0,2)
        self._buf.write(data)
        self._buf.seek(read_pos)
    
    def tell(self):
        v = self._buf.tell()
        return v

    def seek(self,offset,whence):
        self._buf.seek(offset,whence)