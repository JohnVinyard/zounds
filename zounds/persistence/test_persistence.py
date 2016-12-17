import unittest2
from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.timeseries import TimeDimension, Seconds, Milliseconds
from zounds.spectral import FrequencyDimension, FrequencyScale, FrequencyBand
from arraywithunits import ArrayWithUnitsEncoder, ArrayWithUnitsDecoder
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

    def test_can_round_trip_1d_identity_dimension(self):
        raw = np.arange(10)
        arr = ArrayWithUnits(raw, (IdentityDimension(),))
        decoded = self._roundtrip(arr)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(1, len(decoded.dimensions))
        d = decoded.dimensions[0]
        self.assertIsInstance(d, IdentityDimension)
        np.testing.assert_allclose(decoded, raw)

    def test_can_round_trip_2d_with_identity_dimension(self):
        raw = np.random.random_sample((10, 10))
        dim = TimeDimension(Seconds(1), Milliseconds(500))
        arr = ArrayWithUnits(raw, (IdentityDimension(), dim))
        decoded = self._roundtrip(arr)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(2, len(decoded.dimensions))
        idd = decoded.dimensions[0]
        self.assertIsInstance(idd, IdentityDimension)
        td = decoded.dimensions[1]
        self.assertIsInstance(td, TimeDimension)
        self.assertEqual(Seconds(1), td.frequency)
        self.assertEqual(Milliseconds(500), td.duration)

        np.testing.assert_allclose(decoded, raw)

    def test_can_round_trip_1d_constant_rate_time_series(self):
        dim = TimeDimension(Seconds(1), Milliseconds(500))
        raw = np.arange(10)
        ts = ArrayWithUnits(raw, (dim,))
        decoded = self._roundtrip(ts)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(1, len(decoded.dimensions))
        self.assertIsInstance(decoded.dimensions[0], TimeDimension)
        td = decoded.dimensions[0]
        self.assertEqual(Seconds(1), td.frequency)
        self.assertEqual(Milliseconds(500), td.duration)
        np.testing.assert_allclose(decoded, raw)

    def test_can_round_trip_2d_constant_rate_time_series(self):
        dim1 = TimeDimension(Seconds(1), Milliseconds(500))
        scale = FrequencyScale(FrequencyBand(20, 20000), 100)
        dim2 = FrequencyDimension(scale)
        raw = np.random.random_sample((10, 100))
        ts = ArrayWithUnits(raw, (dim1, dim2))
        decoded = self._roundtrip(ts)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(2, len(decoded.dimensions))
        self.assertIsInstance(decoded.dimensions[0], TimeDimension)
        td = decoded.dimensions[0]
        self.assertIsInstance(td, TimeDimension)
        self.assertEqual(Seconds(1), td.frequency)
        self.assertEqual(Milliseconds(500), td.duration)
        fd = decoded.dimensions[1]
        self.assertIsInstance(fd, FrequencyDimension)
        self.assertEqual(scale, fd.scale)
        np.testing.assert_allclose(decoded, raw)

    def test_can_round_trip_3d_constant_rate_time_series_with_frequency_dim(
            self):
        dim1 = TimeDimension(Seconds(2), Milliseconds(1000))
        dim2 = TimeDimension(Seconds(1), Milliseconds(500))
        scale = FrequencyScale(FrequencyBand(20, 20000), 100)
        dim3 = FrequencyDimension(scale)
        raw = np.random.random_sample((5, 2, 100))
        ts = ArrayWithUnits(raw, (dim1, dim2, dim3))

        decoded = self._roundtrip(ts)
        self.assertIsInstance(decoded, ArrayWithUnits)
        self.assertEqual(3, len(decoded.dimensions))

        td1 = decoded.dimensions[0]
        self.assertIsInstance(td1, TimeDimension)
        self.assertEqual(Seconds(2), td1.frequency)
        self.assertEqual(Milliseconds(1000), td1.duration)

        td2 = decoded.dimensions[1]
        self.assertIsInstance(td2, TimeDimension)
        self.assertEqual(Seconds(1), td2.frequency)
        self.assertEqual(Milliseconds(500), td2.duration)

        fd = decoded.dimensions[2]
        self.assertIsInstance(fd, FrequencyDimension)
        self.assertEqual(scale, fd.scale)
        np.testing.assert_allclose(decoded, raw)

    def test_can_pack_bits(self):
        self.fail()
