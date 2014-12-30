from __future__ import division
import unittest2
import os
from uuid import uuid4
import numpy as np
from pysoundfile import *

from zounds.flow.feature import Feature
from zounds.flow.nmpy import NumpyFeature
from zounds.node.bytestream import ByteStream
from zounds.node.audiostream import AudioStream
from zounds.flow.dependency_injection import Registry
from zounds.flow.data import *
from zounds.flow.model import BaseModel

class AudioStreamTest(unittest2.TestCase):
    
    '''
    These tests illustrate my abuse of libsndfile.  Ideally, it should be OK
    to have a reader that is faster than the writer, i.e., I should not give
    up the first time I read zero bytes.  Rather, I should give up only once
    I've read the expected content length, or an error has occurred.
    
    libsndfile seems to demonstrate different behavior depending on the
    encoding type.  For wav files, it behaves as I'd like.  For flac and
    ogg vorbis files, it gives up the first time a read results in zero bytes.
    
    Can I hack libsndfile to always try until n bytes have been read?
    '''
    
    WAV  = snd_types['WAV']  | snd_subtypes['PCM_16']
    OGG  = snd_types['OGG']  | snd_subtypes['VORBIS']
    FLAC = snd_types['FLAC'] | snd_subtypes['PCM_16']
    Model = None
        
    def setUp(self):
        self._file_path = None
        Registry.register(IdProvider,UuidProvider())
        Registry.register(KeyBuilder,StringDelimitedKeyBuilder())
        Registry.register(Database,InMemoryDatabase())
        Registry.register(DataWriter,DataWriter)
        Registry.register(DataReader,DataReaderFactory())
        
        class Doc(BaseModel):
            raw = Feature(\
              ByteStream, 
              chunksize = 4 * 44100 * 10 * 2, 
              store = True)
            pcm = NumpyFeature(AudioStream, needs = raw, store = True)
        
        AudioStreamTest.Model = Doc
        self._seconds = 400
    def tearDown(self):
        try:
            os.remove(self._file_path)
        except OSError:
            pass
        Registry.clear()
    
    def _create_file(self,fmt):
        filename = uuid4().hex
        self._file_path = '/tmp/{filename}'.format(**locals())
        samples = signal(hz = 440, seconds = self._seconds)
        with SoundFile(\
               self._file_path, 
               mode = write_mode, 
               channels = 2, 
               format = fmt, 
               sample_rate = 44100) as f:
            
            for i in xrange(0,len(samples),44100):
                f.write(samples[i : i + 44100])
    
    def _do_test(self,fmt):
        self._create_file(fmt)
        _id = AudioStreamTest.Model.process(raw = self._file_path)
        doc = AudioStreamTest.Model(_id)
        self.assertEqual(self._seconds,doc.pcm.size / 44100)
        self.assertEqual(os.path.getsize(self._file_path),len(doc.raw.read()))
    
    def test_reads_entirety_of_long_ogg_vorbis_file(self):
        self._do_test(AudioStreamTest.OGG)
    
    def test_reads_entirety_of_long_wav_file(self):
        self._do_test(AudioStreamTest.WAV)
    
    def test_reads_entirety_of_long_flac_file(self):
        self._do_test(AudioStreamTest.FLAC)

def signal(hz = 440,seconds=5.,sr=44100.):
    mono = np.random.random_sample(seconds * sr)
    stereo = np.vstack((mono,mono)).T
    return stereo