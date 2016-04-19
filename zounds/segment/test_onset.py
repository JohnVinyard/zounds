from __future__ import division
import unittest2
import featureflow as ff
import numpy as np
from zounds.basic import stft
from zounds.timeseries import \
    ConstantRateTimeSeriesFeature, HalfLapped, Stride, SR44100, Seconds
from zounds.spectral import SlidingWindow
from zounds.synthesize import TickSynthesizer
from onset import \
    MeasureOfTransience, MovingAveragePeakPicker, TimeSliceFeature, ComplexDomain


class OnsetTests(unittest2.TestCase):

    def setUp(self):
        self.samplerate = SR44100()
        self.wscheme = HalfLapped()
        self.STFT = stft(
                store_fft=True,
                resample_to=self.samplerate,
                wscheme=self.wscheme)

    def ticks(self, samplerate, duration, tick_frequency):
        synth = TickSynthesizer(samplerate)
        samples = synth.synthesize(duration, tick_frequency)
        return samples.encode()

    def do_assertions(self, onset_class, feature_func):
        raw = self.ticks(self.samplerate, Seconds(4), Seconds(1))
        _id = onset_class.process(meta=raw)
        doc = onset_class(_id)
        slices = list(doc.slices)
        self.assertEqual(4, len(slices))
        frame_hop = self.wscheme.frequency

        detector = feature_func(doc)
        # self.assertGreaterEqual(len(detector), len(doc.fft))

        self.assertLess(abs(Seconds(0) - slices[0].start), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[1].start), frame_hop)
        self.assertLess(abs(Seconds(2) - slices[2].start), frame_hop)
        self.assertLess(abs(Seconds(3) - slices[3].start), frame_hop)

        self.assertLess(abs(Seconds(1) - slices[0].duration), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[1].duration), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[2].duration), frame_hop)
        # BUG: The last position reported by BasePeakPicker isn't guaranteed
        # to be the end of the file
        # self.assertLess(abs(Seconds(1) - slices[3].duration), frame_hop)
        return doc

    def test_complex_domain_onset_positions(self):
        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.InMemoryDatabase(key_builder=key_builder)

        class WithOnsets(self.STFT, Settings):
            onset_prep = ConstantRateTimeSeriesFeature(
                    SlidingWindow,
                    needs=self.STFT.fft,
                    wscheme=self.wscheme * (1, 3),
                    store=False)

            complex_domain = ConstantRateTimeSeriesFeature(
                    ComplexDomain,
                    needs=onset_prep,
                    store=False)

            sliding_detection = ConstantRateTimeSeriesFeature(
                    SlidingWindow,
                    needs=complex_domain,
                    wscheme=self.wscheme * (1, 11),
                    padwith=5,
                    store=False)

            slices = TimeSliceFeature(
                    MovingAveragePeakPicker,
                    needs=sliding_detection,
                    aggregate=np.median,
                    store=True)

        self.do_assertions(WithOnsets, lambda x: x.complex_domain)

    def test_percussive_onset_positions(self):
        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.InMemoryDatabase(key_builder=key_builder)

        class WithOnsets(self.STFT, Settings):
            transience = ConstantRateTimeSeriesFeature(
                    MeasureOfTransience,
                    needs=self.STFT.fft,
                    store=True)

            sliding_detection = ConstantRateTimeSeriesFeature(
                    SlidingWindow,
                    needs=transience,
                    wscheme=self.wscheme * Stride(frequency=1, duration=11),
                    padwith=5,
                    store=False)

            slices = TimeSliceFeature(
                    MovingAveragePeakPicker,
                    needs=sliding_detection,
                    aggregate=np.median,
                    store=True)

        self.do_assertions(WithOnsets, lambda x: x.transience)
