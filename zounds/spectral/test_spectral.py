import numpy as np
import unittest2
from zounds.basic import resampled
from zounds.util import simple_in_memory_settings
from zounds.timeseries import \
    SR22050, ConstantRateTimeSeriesFeature, Seconds, Picoseconds
from zounds.timeseries.samplerate import SampleRate
from zounds.synthesize import SineSynthesizer, DCTIVSynthesizer
from zounds.spectral import SlidingWindow, DCTIV


class DCTIVTests(unittest2.TestCase):
    def test_reconstruction(self):

        samplerate = SR22050()
        rs = resampled(resample_to=samplerate)

        window_size = Picoseconds(int(1e12))
        wscheme = SampleRate(window_size, window_size)

        @simple_in_memory_settings
        class Document(rs):
            windowed = ConstantRateTimeSeriesFeature(
                SlidingWindow,
                wscheme=wscheme,
                needs=rs.resampled,
                store=False)

            dct = ConstantRateTimeSeriesFeature(
                DCTIV,
                needs=windowed,
                store=True)

        ss = SineSynthesizer(samplerate)
        audio = ss.synthesize(Seconds(5), [440., 660., 880.])

        _id = Document.process(meta=audio.encode())
        doc = Document(_id)

        ds = DCTIVSynthesizer()
        recon = ds.synthesize(doc.dct)

        self.assertEqual(len(audio), len(recon))
        self.assertEqual(audio.samplerate, recon.samplerate)
        orig_fft = abs(np.fft.rfft(audio))
        recon_fft = abs(np.fft.rfft(recon))
        orig_peaks = set(np.argsort(orig_fft)[:-3])
        recon_peaks = set(np.argsort(recon_fft)[:-3])
        # ensure that the original and reconstruction have the same three
        # spectral peaks
        self.assertEqual(orig_peaks, recon_peaks)

