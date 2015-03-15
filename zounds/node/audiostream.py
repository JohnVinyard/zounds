from flow import Node,ByteStream,Graph
from pysoundfile import SoundFile
from io import BytesIO
import numpy as np
      

class Samples(np.ndarray):

    def __new__(cls, input_array, samplerate = None, channels = None):
        obj = np.asarray(input_array).view(cls)
        obj.samplerate = samplerate
        obj.channels = channels
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.samplerate = getattr(obj, 'samplerate', None)
        self.channels = getattr(obj, 'channels', None)  

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
    
    def _enqueue(self,data,pusher):
        self._cache += data
    
    def _dequeue(self):
        v = self._cache
        self._cache = ''
        return v

    def _get_samples(self):
        samples = self._sf.read(self._chunk_size_samples)
        if self._sum_to_mono:
            samples = samples.sum(axis = 1) * 0.5
        channels = 1 if len(samples.shape) == 1 else samples.shape[1]
        return Samples(\
            samples, samplerate = self._sf.sample_rate, channels = channels)
    
    def _process(self,data):
        b = data
        if self._buf is None:
            self._buf = MemoryBuffer(b.total_length)
        
        self._buf.write(b)
        
        if self._sf is None:
            self._sf = SoundFile(self._buf,virtual_io = True)
        
        if not self._finalized: 
            yield self._get_samples()
            return
        
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