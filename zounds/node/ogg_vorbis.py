from __future__ import division
from flow import Node, Decoder, Feature
from audiostream import MemoryBuffer
from soundfile import *
from byte_depth import chunk_size_samples
from duration import Picoseconds, Seconds
from timeseries import TimeSlice

class OggVorbisWrapper(object):
    
    def __init__(self, flo):
        self._flo = flo
        self._sf = SoundFile(self._flo)
        #self._freq = Microseconds(int((1 / self._sf.samplerate) * 1e6))
        self._freq = Picoseconds(int(1e12)) / self._sf.samplerate
    
    def _n_samples(self, duration):
        return int(duration / self._freq)
    
    def __getitem__(self, timeslice):
        if timeslice == slice(None):
            return self._sf[:]
        
        start_sample = int(timeslice.start / self._freq)
        self._sf.seek(start_sample)
        return self._sf.read(self._n_samples(timeslice.duration))
    
    def iter_chunks(self):
        chunksize = Seconds(1)
        ts = TimeSlice(chunksize)
        sl = self[ts]
        yield sl
        while len(sl) >= self._n_samples(chunksize):
            ts += chunksize
            sl = self[ts]
            yield sl

class OggVorbisDecoder(Decoder):
    
    def __init__(self):
        super(OggVorbisDecoder, self).__init__()
    
    def __call__(self, flo):
        return OggVorbisWrapper(flo)
    
    def __iter__(self, flo):
        yield self(flo)

class OggVorbisFeature(Feature):
    
    def __init__(\
         self, 
         extractor, 
         needs = None, 
         store = False, 
         key = None,
         **extractor_args):
        
        super(OggVorbisFeature, self).__init__(\
               extractor,
               needs = needs,
               store = store,
               decoder = OggVorbisDecoder(),
               key = key,
               **extractor_args)

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