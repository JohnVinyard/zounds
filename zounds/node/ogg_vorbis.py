from flow import Node
from audiostream import MemoryBuffer
from soundfile import *
from byte_depth import chunk_size_samples

class OggVorbis(Node):
    
    def __init__(self, needs = None):
        super(OggVorbis,self).__init__(needs = needs)
        self._in_buf = None
        self._in_sf = None
        self._out_buf = None
        self._out_sf = None
        self._already_ogg = None
        self._chunk_size_samples = None
        
        self._total_samples = 0
        self._total_read_samples = 0
        self._total_bytes_written = 0
    
    def _enqueue(self, data, pusher):
        if self._in_buf is None:
            self._in_buf = MemoryBuffer(data.total_length)
            self._in_buf.write(data)
            self._in_sf = SoundFile(self._in_buf)
            self._already_ogg = 'OGG' in self._in_sf.format
        
        if not self._chunk_size_samples:
            self._chunk_size_samples = chunk_size_samples(self._in_sf, data)
        
        if self._already_ogg:
            super(OggVorbis,self)._enqueue(data, pusher)
            return 
        
        if self._out_buf is None:
            self._out_buf = MemoryBuffer(data.total_length)
            self._out_sf = SoundFile(\
             self._out_buf, 
             format = 'OGG', 
             subtype = 'VORBIS',
             mode = 'w',
             samplerate = self._in_sf.samplerate,
             channels = self._in_sf.channels)
        else:
            self._in_buf.write(data)
        
        self._total_bytes_written += len(data)
    
    def _dequeue(self):
        
        if self._already_ogg:
            return super(OggVorbis,self)._dequeue()
        
        samples = self._in_sf.read(self._chunk_size_samples)
        self._total_read_samples += len(samples)
        factor = 20
        while samples.size:
            self._total_samples += samples.shape[0]
            # KLUDGE: Trying to write too-large chunks to an ogg file seems to
            # cause a segfault in libsndfile
            for i in xrange(0, len(samples), self._in_sf.samplerate * factor):
                self._out_sf.write(samples[i : i + self._in_sf.samplerate * factor])
            samples = self._in_sf.read(self._chunk_size_samples)
            self._total_read_samples += len(samples)
        
        return self._out_buf
    
    def _process_other(self, data):
        if self._finalized:
            self._out_sf.close()
            
        o = data.read(count = -1)
        return o

    def _process_ogg(self, data):
        return data
    
    def _process(self, data):        
        if self._already_ogg:
            yield self._process_ogg(data)
        else:
            yield self._process_other(data)