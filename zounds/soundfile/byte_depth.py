from __future__ import division

_lookup = {
    # https://support.microsoft.com/en-us/kb/89879
    'MS_ADPCM': 1,
    'IMA_ADPCM': 1,
    'PCM_S8': 1,
    'PCM_U8': 1,
    'PCM_16': 2,
    'PCM_24': 3,
    'PCM_32': 4,
    'FLOAT': 4,
    'DOUBLE': 8,
    # this is a guess, erring on the side of caution
    'VORBIS': 3
}


def chunk_size_samples(sf, buf):
    """
    Black magic to account for the fact that libsndfile's behavior varies
    depending on file format when using the virtual io api.

    If you ask for more samples from an ogg or flac file than are available
    at that moment, libsndfile will give you no more samples ever, even if
    more bytes arrive in the buffer later.
    """
    byte_depth = _lookup[sf.subtype]
    channels = sf.channels
    bytes_per_second = byte_depth * sf.samplerate * channels
    secs = len(buf) / bytes_per_second
    secs = max(1, secs - 6)
    return int(secs * sf.samplerate)
