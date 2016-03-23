from __future__ import division

from io import BytesIO

import numpy as np
import unittest2
from soundfile import SoundFile

from zounds.timeseries import \
    TimeSlice, ConstantRateTimeSeriesFeature, AudioSamples, AudioSamplesFeature, \
    SR44100, HalfLapped, Seconds, Milliseconds
from zounds.soundfile import \
    AudioStream, OggVorbis, OggVorbisFeature, Resampler
from zounds.spectral import \
    SlidingWindow, OggVorbisWindowingFunc, FFT, Chroma, BarkBands, BFCC
from zounds.basic import Max
from featureflow import *

windowing_scheme = HalfLapped()
samplerate = SR44100()


class Settings(PersistenceSettings):
    id_provider = UuidProvider()
    key_builder = StringDelimitedKeyBuilder()
    database = InMemoryDatabase(key_builder=key_builder)


class Document(BaseModel, Settings):
    raw = ByteStreamFeature(
            ByteStream,
            chunksize=2 * 44100 * 30 * 2,
            store=True)

    ogg = OggVorbisFeature(
            OggVorbis,
            needs=raw,
            store=True)

    pcm = AudioSamplesFeature(
            AudioStream,
            needs=raw,
            store=True)

    resampled = AudioSamplesFeature(
            Resampler,
            needs=pcm,
            samplerate=samplerate,
            store=True)

    windowed = ConstantRateTimeSeriesFeature(
            SlidingWindow,
            needs=resampled,
            wscheme=windowing_scheme,
            wfunc=OggVorbisWindowingFunc(),
            store=False)

    fft = ConstantRateTimeSeriesFeature(
            FFT,
            needs=windowed,
            store=True)

    chroma = ConstantRateTimeSeriesFeature(
            Chroma,
            needs=fft,
            samplerate=samplerate,
            store=True)

    bark = ConstantRateTimeSeriesFeature(
            BarkBands,
            needs=fft,
            samplerate=samplerate,
            store=True)

    bfcc = ConstantRateTimeSeriesFeature(
            BFCC,
            needs=bark,
            store=True)

    bfcc_sliding_window = ConstantRateTimeSeriesFeature(
            SlidingWindow,
            needs=bfcc,
            wscheme=windowing_scheme * (2, 4),
            store=True)

    bfcc_pooled = ConstantRateTimeSeriesFeature(
            Max,
            needs=bfcc_sliding_window,
            axis=1,
            store=True)


class HasUri(object):
    def __init__(self, uri):
        super(HasUri, self).__init__()
        self.uri = uri


def signal(hz=440, seconds=5., sr=44100.):
    # cycles per sample
    cps = hz / sr
    # total samples
    ts = seconds * sr
    mono = np.sin(np.arange(0, ts * cps, cps) * (2 * np.pi))
    return np.column_stack((mono, mono))


def soundfile(hz=440, seconds=5., sr=44100.):
    bio = BytesIO()
    s = signal(hz, seconds, sr)
    with SoundFile(
            bio,
            mode='w',
            channels=2,
            format='WAV',
            subtype='PCM_16',
            samplerate=int(sr)) as f:
        f.write(s)
    bio.seek(0)
    return s, HasUri(bio)


class IntegrationTests(unittest2.TestCase):
    def setUp(self):
        signal, bio = soundfile(seconds=10.)
        _id = Document.process(raw=bio)
        self.doc = Document(_id)
        self.signal = signal

    def test_pcm_returns_audio_samples(self):
        self.assertIsInstance(self.doc.pcm, AudioSamples)

    def test_pcm_is_summed_to_mono(self):
        self.assertEqual(1, self.doc.pcm.channels)

    def test_resampled_returns_audio_samples(self):
        self.assertIsInstance(self.doc.resampled, AudioSamples)

    def test_ogg_vorbis_iter_chunks_returns_audio_samples(self):
        chunks = list(self.doc.ogg.iter_chunks())
        self.assertTrue(
                all(isinstance(chunk, AudioSamples) for chunk in chunks))

    def test_ogg_vorbis_wrapper_returns_audio_samples(self):
        self.assertIsInstance(self.doc.ogg[:], AudioSamples)

    def test_ogg_wrapper_has_correct_duration_seconds(self):
        self.assertEqual(10, self.doc.ogg.duration_seconds)

    def test_windowed_and_fft_have_same_first_dimension(self):
        self.assertEqual(self.doc.windowed.shape[0], self.doc.fft.shape[0])

    def test_fft_dimension_is_half_of_windowsize(self):
        self.assertEqual(self.doc.windowed.shape[1] // 2, self.doc.fft.shape[1])

    def test_bfcc_sliding_window_has_correct_shape(self):
        self.assertEqual((4, 13), self.doc.bfcc_sliding_window.shape[1:])

    def test_bfcc_sliding_window_has_correct_frequency(self):
        self.assertEqual(
                2,
                self.doc.bfcc_sliding_window.frequency / self.doc.bfcc.frequency)

    def test_bfcc_sliding_window_has_correct_duration(self):
        self.assertEqual(
                5,
                self.doc.bfcc_sliding_window.duration / self.doc.bfcc.frequency)

    def test_bfcc_pooled_has_correct_shape(self):
        self.assertEqual(2, len(self.doc.bfcc_pooled.shape))
        self.assertEqual((13,), self.doc.bfcc_pooled.shape[1:])

    def test_bfcc_pooled_has_correct_frequency(self):
        self.assertEqual(
                2, self.doc.bfcc_pooled.frequency / self.doc.bfcc.frequency)

    def test_bfcc_pooled_has_correct_duration(self):
        self.assertEqual(
                5, self.doc.bfcc_pooled.duration / self.doc.bfcc.frequency)

    def test_can_get_second_long_slice_from_ogg_vorbis_feature(self):
        ogg = self.doc.ogg
        samples = ogg[TimeSlice(Seconds(1), start=Seconds(5))]
        self.assertEqual(44100, len(samples))

    def test_can_get_entirety_of_ogg_vorbis_feature_with_slice(self):
        ogg = self.doc.ogg
        samples = ogg[:]
        self.assertEqual(441000, len(samples))

    def test_can_read_ogg_samples_twice(self):
        ogg = self.doc.ogg
        s1 = ogg[:]
        s2 = ogg[:]
        self.assertEqual(s1.shape, s2.shape)

    def test_can_get_end_of_ogg_vorbis_feature_with_slice(self):
        ogg = self.doc.ogg
        samples = ogg[TimeSlice(Seconds(1), Milliseconds(9500))]
        self.assertEqual(22050, len(samples))

    def test_sliding_window_has_correct_relationship_to_bfcc(self):
        self.assertEqual(
                2,
                self.doc.bfcc.shape[0] // self.doc.bfcc_sliding_window.shape[0])
