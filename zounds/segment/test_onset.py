
import unittest2
import featureflow as ff
import numpy as np

from zounds.util import simple_in_memory_settings
from zounds.basic import stft, Pooled
from zounds.timeseries import \
    HalfLapped, Stride, SR44100, Seconds, VariableRateTimeSeriesFeature
from zounds.spectral import SlidingWindow
from zounds.synthesize import TickSynthesizer
from .onset import \
    MeasureOfTransience, MovingAveragePeakPicker, TimeSliceFeature, \
    ComplexDomain
from zounds.persistence import ArrayWithUnitsFeature


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
        print('DURATION IN SECONDS', samples.span, samples.shape)
        return samples.encode()

    def do_assertions(self, onset_class, feature_func):
        raw = self.ticks(self.samplerate, Seconds(4), Seconds(1))
        _id = onset_class.process(meta=raw)
        doc = onset_class(_id)
        slices = list(doc.slices.slices)
        self.assertEqual(4, len(slices))
        frame_hop = self.wscheme.frequency

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

    @unittest2.skip
    def test_complex_domain_onset_positions(self):
        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.InMemoryDatabase(key_builder=key_builder)

        class WithOnsets(self.STFT, Settings):
            onset_prep = ArrayWithUnitsFeature(
                SlidingWindow,
                needs=self.STFT.fft,
                wscheme=self.wscheme * (1, 3),
                store=False)

            complex_domain = ArrayWithUnitsFeature(
                ComplexDomain,
                needs=onset_prep,
                store=False)

            sliding_detection = ArrayWithUnitsFeature(
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
        @simple_in_memory_settings
        class WithOnsets(self.STFT):
            transience = ArrayWithUnitsFeature(
                MeasureOfTransience,
                needs=self.STFT.fft,
                store=True)

            sliding_detection = ArrayWithUnitsFeature(
                SlidingWindow,
                needs=transience,
                wscheme=self.wscheme * Stride(frequency=1, duration=10),
                padwith=5,
                store=False)

            slices = TimeSliceFeature(
                MovingAveragePeakPicker,
                needs=sliding_detection,
                aggregate=np.median,
                store=True)

        self.do_assertions(WithOnsets, lambda x: x.transience)

    def test_can_pool_stored_time_slice_feature(self):
        @simple_in_memory_settings
        class WithOnsets(self.STFT):
            transience = ArrayWithUnitsFeature(
                MeasureOfTransience,
                needs=self.STFT.fft,
                store=True)

            sliding_detection = ArrayWithUnitsFeature(
                SlidingWindow,
                needs=transience,
                wscheme=self.wscheme * Stride(frequency=1, duration=10),
                padwith=5,
                store=False)

            slices = TimeSliceFeature(
                MovingAveragePeakPicker,
                needs=sliding_detection,
                aggregate=np.median,
                store=True)

            pooled = VariableRateTimeSeriesFeature(
                Pooled,
                needs=(self.STFT.fft, slices),
                op=np.max,
                axis=0,
                store=True)

        signal = self.ticks(self.samplerate, Seconds(4), Seconds(1))
        _id = WithOnsets.process(meta=signal)
        doc = WithOnsets(_id)
        self.assertEqual((4, 1025), doc.pooled.slicedata.shape)

    def test_can_pool_non_stored_time_slice_feature(self):
        @simple_in_memory_settings
        class WithOnsets(self.STFT):
            transience = ArrayWithUnitsFeature(
                MeasureOfTransience,
                needs=self.STFT.fft,
                store=True)

            sliding_detection = ArrayWithUnitsFeature(
                SlidingWindow,
                needs=transience,
                wscheme=self.wscheme * Stride(frequency=1, duration=10),
                padwith=5,
                store=False)

            slices = TimeSliceFeature(
                MovingAveragePeakPicker,
                needs=sliding_detection,
                aggregate=np.median,
                store=True)

        signal = self.ticks(self.samplerate, Seconds(4), Seconds(1))
        _id = WithOnsets.process(meta=signal)

        class WithPooled(WithOnsets):
            pooled = VariableRateTimeSeriesFeature(
                Pooled,
                needs=(self.STFT.fft, WithOnsets.slices),
                op=np.max,
                axis=0,
                store=True)

        doc = WithPooled(_id)
        self.assertEqual((4, 1025), doc.pooled.slicedata.shape)
