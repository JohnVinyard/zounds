import unittest2
from audiograph import resampled, stft
from zounds.timeseries.samplerate import SR11025, SR22050
from zounds.timeseries.duration import Seconds
from zounds.util.persistence import simple_in_memory_settings
from zounds.synthesize.synthesize import NoiseSynthesizer, SineSynthesizer
import zipfile
from io import BytesIO
import featureflow


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


class StftTests(unittest2.TestCase):

    def test_can_hadle_zip_file(self):
        STFT = stft(resample_to=SR11025(), store_fft=True)

        @simple_in_memory_settings
        class Document(STFT):
            pass

        signal = SineSynthesizer(SR11025()).synthesize(Seconds(2))

        bio = BytesIO()
        filename = 'test.wav'
        with zipfile.ZipFile(bio, mode='w') as zf:
            zf.writestr(filename, signal.encode().read())
        bio.seek(0)

        zip_wrapper = featureflow.iter_zip(bio).next()
        _id = Document.process(meta=zip_wrapper)
        doc = Document(_id)
        self.assertEqual(2, doc.ogg.duration_seconds)
