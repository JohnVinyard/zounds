import numpy as np
import unittest2
from zounds.basic import resampled
from zounds.util import simple_in_memory_settings
from zounds.timeseries import \
    SR11025, SR22050, ConstantRateTimeSeriesFeature, Seconds, Picoseconds
from zounds.timeseries.samplerate import SampleRate, HalfLapped
from zounds.synthesize import SineSynthesizer, DCTIVSynthesizer, MDCTSynthesizer
from zounds.spectral import SlidingWindow, DCTIV, MDCT
from tfrepresentation import \
    TimeFrequencyRepresentation, TimeFrequencyRepresentationFeature


class MDCTTests(unittest2.TestCase):
    def setUp(self):
        self.samplerate = SR11025()
        rs = resampled(resample_to=self.samplerate)

        wscheme = HalfLapped()

        @simple_in_memory_settings
        class Document(rs):
            windowed = ConstantRateTimeSeriesFeature(
                    SlidingWindow,
                    wscheme=wscheme,
                    needs=rs.resampled,
                    store=False)

            mdct = TimeFrequencyRepresentationFeature(
                    MDCT,
                    needs=windowed,
                    store=True)

        ss = SineSynthesizer(self.samplerate)
        self.audio = ss.synthesize(Seconds(5), [440., 660., 880.])

        _id = Document.process(meta=self.audio.encode())
        self.doc = Document(_id)

    def test_is_correct_type(self):
        self.assertIsInstance(self.doc.mdct, TimeFrequencyRepresentation)

    def test_has_correct_nyquist_frequency(self):
        self.assertEqual(self.samplerate.nyquist, self.doc.mdct.scale.stop_hz)

    @unittest2.skip
    def test_reconstruction(self):
        ds = MDCTSynthesizer()
        recon = ds.synthesize(self.doc.mdct)

        # self.assertEqual(len(self.audio), len(recon))
        self.assertEqual(self.audio.samplerate, recon.samplerate)
        orig_fft = abs(np.fft.rfft(self.audio))
        recon_fft = abs(np.fft.rfft(recon))
        orig_peaks = set(np.argsort(orig_fft)[-3:])
        recon_peaks = set(np.argsort(recon_fft)[-3:])
        print len(orig_peaks), len(recon_peaks)
        # ensure that the original and reconstruction have the same three
        # spectral peaks
        self.assertEqual(orig_peaks, recon_peaks)


class DCTIVTests(unittest2.TestCase):
    def setUp(self):
        self.samplerate = SR22050()
        rs = resampled(resample_to=self.samplerate)

        window_size = Picoseconds(int(1e12))
        wscheme = SampleRate(window_size, window_size)

        @simple_in_memory_settings
        class Document(rs):
            windowed = ConstantRateTimeSeriesFeature(
                    SlidingWindow,
                    wscheme=wscheme,
                    needs=rs.resampled,
                    store=False)

            dct = TimeFrequencyRepresentationFeature(
                    DCTIV,
                    needs=windowed,
                    store=True)

        ss = SineSynthesizer(self.samplerate)
        self.audio = ss.synthesize(Seconds(5), [440., 660., 880.])

        _id = Document.process(meta=self.audio.encode())
        self.doc = Document(_id)

    def test_is_correct_type(self):
        self.assertIsInstance(self.doc.dct, TimeFrequencyRepresentation)

    def test_has_correct_nyquist_frequency(self):
        self.assertEqual(self.samplerate.nyquist, self.doc.dct.scale.stop_hz)

    def test_reconstruction(self):
        ds = DCTIVSynthesizer()
        recon = ds.synthesize(self.doc.dct)

        self.assertEqual(len(self.audio), len(recon))
        self.assertEqual(self.audio.samplerate, recon.samplerate)
        orig_fft = abs(np.fft.rfft(self.audio))
        recon_fft = abs(np.fft.rfft(recon))
        orig_peaks = set(np.argsort(orig_fft)[-3:])
        recon_peaks = set(np.argsort(recon_fft)[-3:])
        # ensure that the original and reconstruction have the same three
        # spectral peaks
        self.assertEqual(orig_peaks, recon_peaks)
