import unittest2
from audiograph import resampled
from zounds.timeseries.samplerate import SR11025, SR22050
from zounds.timeseries.duration import Seconds
from zounds.util.persistence import simple_in_memory_settings
from zounds.synthesize.synthesize import NoiseSynthesizer


class ResampledTests(unittest2.TestCase):
    def test_audio_is_resampled(self):

        orig_sample_rate = SR22050()
        new_sample_rate = SR11025()

        rs = resampled(resample_to=new_sample_rate, store_resampled=True)

        @simple_in_memory_settings
        class Document(rs):
            pass

        synth = NoiseSynthesizer(orig_sample_rate)
        samples = synth.synthesize(Seconds(3))
        _id = Document.process(meta=samples.encode())
        doc = Document(_id)
        self.assertEqual(new_sample_rate, doc.resampled.samplerate)
        self.assertEqual(len(samples) // 2, len(doc.resampled))
