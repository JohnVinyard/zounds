# from __future__ import division
#
# import os
# from io import BytesIO
# from random import choice
# from uuid import uuid4
#
# import numpy as np
# import unittest2
# from soundfile import *
#
# from zounds.soundfile import AudioStream, OggVorbis, Resampler
# from zounds.timeseries import SR44100
# from featureflow import \
#     BaseModel, ByteStream, ByteStreamFeature, Feature, UuidProvider, \
#     InMemoryDatabase, StringDelimitedKeyBuilder, PersistenceSettings
# from featureflow.nmpy import NumpyFeature
#
# _sample_rates = (11025, 22050, 44100, 48000, 88200, 96000)
# _channels = (1, 2)
# _formats = (
#     ('WAV', 'PCM_16'),
#     ('WAV', 'PCM_24'),
#     ('WAV', 'PCM_32'),
#     ('WAV', 'FLOAT'),
#     ('WAV', 'DOUBLE'),
#
#     ('AIFF', 'PCM_16'),
#     ('AIFF', 'PCM_24'),
#     ('AIFF', 'PCM_32'),
#     ('AIFF', 'FLOAT'),
#     ('AIFF', 'DOUBLE'),
#
#     ('FLAC', 'PCM_16'),
#     ('FLAC', 'PCM_24'),
#
#     ('OGG', 'VORBIS')
# )
#
#
# class FuzzTests(unittest2.TestCase):
#     def __init__(
#             self, chunksize_bytes, samplerate, fmt, subtype, channels, seconds):
#
#         super(FuzzTests, self).__init__()
#         self._samplerate = samplerate
#         self._fmt = fmt
#         self._subtype = subtype
#         self._seconds = seconds
#         self._channels = channels
#         self._chunksize_bytes = chunksize_bytes
#
#     def __repr__(self):
#         return 'FuzzTests(sr = {_samplerate}, fmt = {_fmt}, st = {_subtype}, secs = {_seconds}, ch = {_channels}, cs = {_chunksize_bytes})'.format(
#                 **self.__dict__)
#
#     def __str__(self):
#         return self.__repr__()
#
#     def model_cls(self):
#
#         class Settings(PersistenceSettings):
#             id_provider = UuidProvider()
#             key_builder = StringDelimitedKeyBuilder()
#             database = InMemoryDatabase(key_builder=key_builder)
#
#         class Document(BaseModel, Settings):
#             raw = ByteStreamFeature(
#                     ByteStream,
#                     chunksize=self._chunksize_bytes,
#                     store=False)
#
#             ogg = Feature(
#                     OggVorbis,
#                     needs=raw,
#                     store=True)
#
#             pcm = NumpyFeature(
#                     AudioStream,
#                     needs=raw,
#                     store=True)
#
#             resampled = NumpyFeature(
#                     Resampler,
#                     samplerate=SR44100(),
#                     needs=pcm,
#                     store=True)
#
#         return Document
#
#     def runTest(self):
#         # TODO: Update this to use BytesIO instead of writing a file to disk
#         self._fn = '/tmp/' + uuid4().hex
#         print self
#
#         n_samples = int(self._samplerate * self._seconds)
#         samples = np.sin(np.arange(0, n_samples * 440, 440) * (2 * np.pi))
#         if self._channels == 2:
#             samples = np.repeat(samples, 2).reshape((n_samples, 2))
#
#         try:
#             with SoundFile(
#                     self._fn,
#                     mode='w',
#                     samplerate=self._samplerate,
#                     channels=self._channels,
#                     format=self._fmt,
#                     subtype=self._subtype) as sf:
#                 for i in range(0, n_samples, 44100):
#                     sf.write(samples[i: i + 44100])
#         except ValueError as e:
#             self.fail(e)
#
#         class HasUri(object):
#             def __init__(self, uri):
#                 self.uri = uri
#
#         model = self.model_cls()
#         _id = model.process(raw=HasUri(self._fn))
#         doc = model(_id)
#         orig_samples = doc.pcm
#         self.assertAlmostEqual(
#                 samples.shape[0], orig_samples.shape[0], delta=1)
#
#         del orig_samples
#         resampled = doc.resampled
#         seconds = resampled.shape[0] / 44100
#         self.assertAlmostEqual(self._seconds, seconds, delta=.025)
#         del resampled
#         ogg_bytes = doc.ogg
#
#         # first, do the ogg conversion "by hand" to make sure I'm not missing
#         # something
#         bio = BytesIO()
#         with SoundFile( \
#                 bio,
#                 format='OGG',
#                 subtype='VORBIS',
#                 mode='w',
#                 samplerate=self._samplerate,
#                 channels=self._channels) as ogg_sf:
#             for i in xrange(0, n_samples, 44100):
#                 ogg_sf.write(samples[i: i + 44100])
#
#         bio.seek(0)
#         ogg_bytes.seek(0)
#         bio.seek(0)
#
#         with SoundFile(ogg_bytes) as ogg_sf:
#             ogg_samples = ogg_sf.read(samples.shape[0] + 99999)
#
#         ogg_seconds = ogg_samples.shape[0] / self._samplerate
#         self.assertAlmostEqual(self._seconds, ogg_seconds, delta=.025)
#
#     def tearDown(self):
#         os.remove(self._fn)
#
#
# def suite():
#     suite = unittest2.TestSuite()
#
#     for _ in xrange(50):
#         seconds = (np.random.random_sample() * 50)
#         min_size = 4 * 96000 * 5 * 2
#         chunksize = min_size + (np.random.randint(0, 4 * 96000 * 25 * 2))
#         samplerate = choice(_sample_rates)
#         fmt = choice(_formats)
#         channels = choice(_channels)
#         suite.addTest(
#             FuzzTests(chunksize, samplerate, fmt[0], fmt[1], channels,
#                       seconds))
#
#     return suite
#
#
# if __name__ == '__main__':
#     unittest2.TextTestRunner().run(suite())
