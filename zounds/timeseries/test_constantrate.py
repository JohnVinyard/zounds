import unittest2
from constantrate import ConstantRateTimeSeries
from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.timeseries import TimeDimension, Seconds, Milliseconds, TimeSlice
import numpy as np


class ConstantRateTimeSeriesTests(unittest2.TestCase):
    def test_raises_when_not_array_with_units_instance(self):
        arr = np.zeros(10)
        self.assertRaises(ValueError, lambda: ConstantRateTimeSeries(arr))

    def test_raises_when_first_dimension_is_not_time_dimension(self):
        raw = np.zeros((10, 3))
        arr = ArrayWithUnits(raw, dimensions=[
            IdentityDimension(), TimeDimension(frequency=Seconds(1))])
        self.assertRaises(ValueError, lambda: ConstantRateTimeSeries(arr))

    def test_iter_slices_yields_evenly_spaced_time_slices(self):
        raw = np.random.random_sample((10, 3))
        arr = ArrayWithUnits(raw, dimensions=[
            TimeDimension(frequency=Milliseconds(500), duration=Seconds(1)),
            IdentityDimension()
        ])
        crts = ConstantRateTimeSeries(arr)
        slices = list(crts.iter_slices())
        self.assertEqual(10, len(slices))

        ts1, d1 = slices[0]
        self.assertEqual(
            TimeSlice(start=Seconds(0), duration=Seconds(1)), ts1)
        np.testing.assert_allclose(raw[0], d1)

        ts2, d2 = slices[1]
        self.assertEqual(
            TimeSlice(start=Milliseconds(500), duration=Seconds(1)), ts2)
        np.testing.assert_allclose(raw[1], d2)
