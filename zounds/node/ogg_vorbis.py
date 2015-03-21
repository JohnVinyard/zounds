from flow import Node
from audiostream import MemoryBuffer
from soundfile import *

class OggVorbis(Node):
    
    def __init__(self, needs = None):
        super(OggVorbis,self).__init__(needs = needs)
        self._in_buf = None
        self._in_sf = None
        self._out_buf = None
        self._out_sf = None
        self._already_ogg = None
    
    def _process_ogg(self,data):
        return data
    
    def _process_other(self,data):
        
        if self._out_buf is None:
            self._in_buf = MemoryBuffer(data.total_length)
            self._in_buf.write(data)
            self._in_sf = SoundFile(self._in_buf)
            self._out_buf = MemoryBuffer(data.total_length)
            
            self._out_sf = SoundFile(\
             self._out_buf, 
             format = 'OGG', 
             mode = 'w',
             samplerate = self._in_sf.samplerate,
             channels = self._in_sf.channels)
        else:
            self._in_buf.write(data)        
        
        samples = self._in_sf.read()
        self._out_sf.write(samples)
        
        if self._finalized:
            self._out_sf.flush()
            
        return self._out_buf.read(count = -1)
    
    def _process(self, data):
        if self._in_buf is None:
            self._in_buf = MemoryBuffer(data.total_length)
            self._in_buf.write(data)
            self._in_sf = SoundFile(self._in_buf)
            self._already_ogg = 'OGG' in self._in_sf.format
        
        if self._already_ogg:
            yield self._process_ogg(data)
        else:
            yield self._process_other(data)