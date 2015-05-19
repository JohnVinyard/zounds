from __future__ import division

_lookup = {
       'PCM_16' : 2,
       'PCM_24' : 3,
       'PCM_32' : 4,
       'FLOAT'  : 4,
       'DOUBLE' : 8,
       # this is a guess, erring on the side of caution
       'VORBIS' : 3
}

#def get_byte_depth(subtype):
#    return _lookup[subtype]

def chunk_size_samples(sf, buf):
    #print 'BUF',len(buf)
    byte_depth = _lookup[sf.subtype]
    #print 'DEPTH',byte_depth
    channels = 2
    bytes_per_second = byte_depth * sf.samplerate * channels
    #print 'BYTES PER SECOND',bytes_per_second
    secs = len(buf) / bytes_per_second
    #print 'SECS',secs
    #print 'SAMPLERATE',sf.samplerate
    secs = secs if secs <= 4 else secs - 4
    #print 'SECS',secs
    css = int(secs * sf.samplerate)
    #print 'CHUNK SIZE SAMPLES', css
    return css
    
        
