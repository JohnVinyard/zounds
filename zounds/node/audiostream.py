from flow import Node,ByteStream,Graph
from pysoundfile import SoundFile
from io import BytesIO

class AudioStream(Node):
    
    def __init__(\
         self, 
         sum_to_mono = True, 
         chunk_size_samples = 44100 * 20, 
         needs = None):
        
        super(AudioStream, self).__init__(needs = needs)
        self._sum_to_mono = sum_to_mono
        self._buf = None
        self._sf = None
        self._chunk_size_samples = chunk_size_samples
        self._cache = ''
        
        self._total_samples = 0
    
    def _enqueue(self,data,pusher):
        self._cache += data
    
    def _dequeue(self):
        v = self._cache
        self._cache = ''
        return v

    def _get_samples(self):
        samples = self._sf.read(self._chunk_size_samples)
        print 'samples size',len(samples)
        self._total_samples += len(samples)
        print 'total samples',self._total_samples
        if self._sum_to_mono:
            samples = samples.sum(axis = 1) * 0.5
        return samples
    
    def _process(self,data):
        b = data
        print 'bytes read',len(b)
        if self._buf is None:
            self._buf = MemoryBuffer(b.total_length)
        
        self._buf.write(b)
        
        if self._sf is None:
            self._sf = SoundFile(self._buf,virtual_io = True)
        
        if not self._finalized: 
            yield self._get_samples()
            return
        
        print 'FINALIZED'
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