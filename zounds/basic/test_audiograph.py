import unittest2
from .audiograph import resampled, stft, frequency_adaptive
from zounds.timeseries.samplerate import \
    SR11025, SR22050, SampleRate, nearest_audio_sample_rate, HalfLapped
from zounds.timeseries.duration import Seconds, Milliseconds
from zounds.util.persistence import simple_in_memory_settings
from zounds.persistence import ArrayWithUnitsFeature
from zounds.synthesize.synthesize import NoiseSynthesizer, SineSynthesizer
from zounds.spectral import GeometricScale, FrequencyAdaptive
import zipfile
from io import BytesIO
import featureflow
import numpy as np


class FrequencyAdaptiveTests(unittest2.TestCase):
    def test_can_compute_frequency_adaptive_feature(self):
        scale = GeometricScale(
            start_center_hz=50,
            stop_center_hz=5000,
            bandwidth_ratio=0.07123,
            n_bands=128)

        fa = frequency_adaptive(
            SampleRate(frequency=Milliseconds(358), duration=Milliseconds(716)),
            scale,
            store_freq_adaptive=True)

        @simple_in_memory_settings
        class Document(fa):
            rasterized = ArrayWithUnitsFeature(
                lambda fa: fa.rasterize(64).astype(np.float32),
                needs=fa.freq_adaptive,
                store=True)

        synth = SineSynthesizer(SR22050())
        samples = synth.synthesize(Seconds(10))
        _id = Document.process(meta=samples.encode())
        doc = Document(_id)
        self.assertIsInstance(doc.freq_adaptive, FrequencyAdaptive)
        self.assertEqual((64, 128), doc.rasterized.shape[-2:])


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

    def test_can_pass_padding_samples(self):
        samplerate = SR11025()

        STFT = stft(
            resample_to=samplerate,
            store_fft=True,
            fft_padding_samples=1024)

        @simple_in_memory_settings
        class Document(STFT):
            pass

        samples = SineSynthesizer(samplerate).synthesize(Seconds(2))
        _id = Document.process(meta=samples.encode())
        doc = Document(_id)
        stft_window_duration = int(np.round(
            HalfLapped().duration / samplerate.frequency))
        expected_samples = ((stft_window_duration + 1024) // 2) + 1
        self.assertEqual(expected_samples, doc.fft.shape[-1])

    def test_can_handle_zip_file(self):
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

        with list(featureflow.iter_zip(bio))[0] as zip_wrapper:
            _id = Document.process(meta=zip_wrapper)
            doc = Document(_id)
            self.assertEqual(2, doc.ogg.duration_seconds)
