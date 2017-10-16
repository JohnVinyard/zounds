import unittest2
from ogg_vorbis import OggVorbisWrapper
from zounds.timeseries import TimeSlice, Seconds, SR11025
from zounds.synthesize import SineSynthesizer


class TestOggVorbisWrapper(unittest2.TestCase):
    def test_can_apply_empty_time_slice_to_wrapper(self):
        synth = SineSynthesizer(SR11025())
        samples = synth.synthesize(Seconds(10))
        encoded = samples.encode(fmt='OGG', subtype='VORBIS')
        wrapper = OggVorbisWrapper(encoded)
        samples = wrapper[TimeSlice()]
        expected = Seconds(10) / Seconds(1)
        actual = samples.end / Seconds(1)
        self.assertAlmostEqual(expected, actual, places=6)

    def test_can_apply_open_ended_slice_to_wrapper(self):
        synth = SineSynthesizer(SR11025())
        samples = synth.synthesize(Seconds(10))
        encoded = samples.encode(fmt='OGG', subtype='VORBIS')
        wrapper = OggVorbisWrapper(encoded)
        samples = wrapper[TimeSlice(start=Seconds(1))]
        expected = Seconds(9) / Seconds(1)
        actual = samples.end / Seconds(1)
        self.assertAlmostEqual(expected, actual, places=6)
