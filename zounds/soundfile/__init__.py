"""
The soundfile module introduces :class:`featureflow.Node` subclasses that know
how to process low-level audio samples and common audio encodings.
"""

from audio_metadata import MetaData, AudioMetaDataEncoder, FreesoundOrgConfig

from ogg_vorbis import \
    OggVorbis, OggVorbisDecoder, OggVorbisEncoder, OggVorbisFeature, \
    OggVorbisWrapper

from audiostream import AudioStream

from resample import Resampler

from chunksize import ChunkSizeBytes
