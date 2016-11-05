import unittest2
from axis import ArrayWithUnits
from zounds.timeseries import \
    TimeDimension, SR44100, Seconds, Milliseconds
from zounds.spectral import FrequencyDimension, FrequencyScale, FrequencyBand
from persistence import ArrayWithUnitsEncoder, ArrayWithUnitsDecoder
import numpy as np
from io import BytesIO


class ArrayWithUnitsFeatureTests(unittest2.TestCase):

    def _roundtrip(self, arr):
        encoder = ArrayWithUnitsEncoder()
        decoder = ArrayWithUnitsDecoder()
        encoded = BytesIO(''.join(encoder._process(arr)))
        return decoder(encoded)

    def test_can_round_trip_audio_samples(self):
        self.fail()

    def test_can_round_trip_1d_constant_rate_time_series(self):
        dim = TimeDimension(Seconds(1), Milliseconds(500))
        ts = ArrayWithUnits(np.arange(10), (dim, ))
        decoded = self._roundtrip(ts)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(1, len(decoded.dimensions))
        self.assertIsInstance(decoded.dimensions[0], TimeDimension)
        td = decoded.dimensions[0]
        self.assertEqual(Seconds(1), td.frequency)
        self.assertEqual(Milliseconds(500), td.duration)

    def test_can_round_trip_2d_constant_rate_time_series(self):
        dim1 = TimeDimension(Seconds(1), Milliseconds(500))
        scale = FrequencyScale(FrequencyBand(20, 20000), 100)
        dim2 = FrequencyDimension(scale)
        ts = ArrayWithUnits(np.random.random_sample((10, 100)), (dim1, dim2))
        decoded = self._roundtrip(ts)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(2, len(decoded.dimensions))
        self.assertIsInstance(decoded.dimensions[0], TimeDimension)
        td = decoded.dimensions[0]
        self.assertIsInstance(td, FrequencyDimension)
        self.assertEqual(Seconds(1), td.frequency)
        self.assertEqual(Milliseconds(500), td.duration)
        fd = decoded.dimensions[1]
        self.assertIsInstance(fd, FrequencyDimension)
        self.assertEqual(scale, fd.scale)

    def test_can_round_trip_3d_constant_rate_time_series_with_frequency_dim(
            self):
        self.fail()

    def test_can_pack_bits(self):
        self.fail()
