from __future__ import division

from flow import *
from flow.nmpy import *
from io import BytesIO
import unittest2

from zounds.node.audiostream import AudioStream
from zounds.node.ogg_vorbis import OggVorbis, OggVorbisFeature
from zounds.node.resample import Resampler
from zounds.node.spectral import FFT, Chroma
from zounds.node.sliding_window import SlidingWindow, OggVorbisWindowingFunc
from zounds.node.timeseries import ConstantRateTimeSeriesFeature
from zounds.node.samplerate import SR44100, HalfLapped

from soundfile import SoundFile
import numpy as np

windowing_scheme = HalfLapped()
samplerate = SR44100()

class Document(BaseModel):

    raw = ByteStreamFeature(\
        ByteStream, 
        chunksize = 2 * 44100 * 30 * 2,
        store = False)

    ogg = OggVorbisFeature(\
        OggVorbis,
        needs = raw,
        store = True)

    pcm = ConstantRateTimeSeriesFeature(\
        AudioStream, 
        needs = raw, 
        store = False)

    resampled = ConstantRateTimeSeriesFeature(\
        Resampler,
        needs = pcm,
        samplerate = samplerate,
        store = True)

    windowed = ConstantRateTimeSeriesFeature(\
        SlidingWindow, 
        needs = resampled,
        wscheme = windowing_scheme,
        wfunc = OggVorbisWindowingFunc(),
        store = False) 

    fft = ConstantRateTimeSeriesFeature(\
        FFT,
        needs = windowed,
        store = True)
    
    chroma = ConstantRateTimeSeriesFeature(\
        Chroma,
        needs = fft,
        samplerate = samplerate,
        store = True)

class HasUri(object):
    
    def __init__(self, uri):
        super(HasUri, self).__init__()
        self.uri = uri

def signal(hz = 440,seconds = 5.,sr = 44100.):
    mono = np.random.random_sample(seconds * sr)
    stereo = np.vstack((mono,mono)).T
    return stereo

def soundfile(hz = 440, seconds = 5., sr = 44100.):
    bio = BytesIO()
    s = signal(hz, seconds, sr)
    with SoundFile(\
       bio, 
       mode = 'w', 
       channels = 2, 
       format = 'WAV',
       subtype = 'PCM_16', 
       samplerate = int(sr)) as f:
        for i in xrange(0, len(s), sr):
            f.write(s[i : i + sr])
    return HasUri(bio)

class IntegrationTests(unittest2.TestCase):
    
    def setUp(self):
        Registry.register(IdProvider, UuidProvider())
        Registry.register(KeyBuilder, StringDelimitedKeyBuilder())
        Registry.register(Database, InMemoryDatabase())
        Registry.register(DataWriter, DataWriter)
    
    def test_windowed_and_fft_have_same_first_dimension(self):
        bio = soundfile(seconds = 10.)
        _id = Document.process(raw = bio)
        doc = Document(_id)
        print 'COMPUTING'
        print 'rehydrated', doc.resampled.shape, doc.resampled.frequency, doc.resampled.duration
        self.assertEqual(doc.windowed.shape[0], doc.fft.shape[0])
    
    
    