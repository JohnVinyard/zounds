from __future__ import division
import unittest2
import featureflow as ff
import numpy as np
from zounds.basic import stft
from zounds.timeseries import \
    ConstantRateTimeSeriesFeature, HalfLapped, Stride, SR44100, Seconds
from zounds.spectral import SlidingWindow
from onset import \
    MeasureOfTransience, MovingAveragePeakPicker, TimeSliceFeature
from soundfile import SoundFile
from io import BytesIO


class OnsetTests(unittest2.TestCase):

    def ticks(self, samplerate, duration, tick_frequency):
        sr = samplerate.samples_per_second
        # create a short, tick sound
        tick = np.random.random_sample(int(sr * .05))
        tick *= np.linspace(1, 0, len(tick))
        # create silence
        samples = np.zeros(sr * (duration / Seconds(1)))
        ticks_per_second = Seconds(1) / tick_frequency
        # introduce periodic ticking sound
        step = int(sr // ticks_per_second)
        for i in xrange(0, len(samples), step):
            samples[i:i+len(tick)] = tick
        # write the samples to a file-like object
        bio = BytesIO()
        with SoundFile(
                bio,
                mode='w',
                channels=1,
                format='WAV',
                subtype='PCM_16',
                samplerate=sr) as f:
            f.write(samples)
        bio.seek(0)
        return bio

    @unittest2.skip
    def test_onset_positions(self):
        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.InMemoryDatabase(key_builder=key_builder)

        samplerate = SR44100()
        wscheme = HalfLapped()
        STFT = stft(store_fft=True, resample_to=samplerate, wscheme=wscheme)

        class WithOnsets(STFT, Settings):
            transience = ConstantRateTimeSeriesFeature(
                MeasureOfTransience,
                needs=STFT.fft,
                store=True)

            sliding_detection = ConstantRateTimeSeriesFeature(
                SlidingWindow,
                needs=transience,
                wscheme=HalfLapped() * Stride(frequency=1, duration=11),
                padwith=5,
                store=False)

            slices = TimeSliceFeature(
                MovingAveragePeakPicker,
                needs=sliding_detection,
                aggregate=np.median,
                store=True)

        raw = self.ticks(samplerate, Seconds(4), Seconds(1))
        _id = WithOnsets.process(meta=raw)
        doc = WithOnsets(_id)
        slices = list(doc.slices)
        self.assertEqual(4, len(slices))
        frame_hop = wscheme.frequency

        frames = np.array(map(lambda x: x.start, slices)) / frame_hop
        print frames
        print np.diff(frames)

        self.assertLess(abs(Seconds(0) - slices[0].start), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[1].start), frame_hop)
        self.assertLess(abs(Seconds(2) - slices[2].start), frame_hop)
        self.assertLess(abs(Seconds(3) - slices[3].start), frame_hop)

        self.assertLess(abs(Seconds(1) - slices[0].duration), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[1].duration), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[2].duration), frame_hop)
        self.assertLess(abs(Seconds(1) - slices[3].duration), frame_hop)

