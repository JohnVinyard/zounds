import unittest2
from zounds.core import ArrayWithUnits
from zounds.timeseries import TimeDimension, Seconds, VariableRateTimeSeries
from onset import BasePeakPicker
import numpy as np


class BasePeakPickerTests(unittest2.TestCase):

    class PeakPicker(BasePeakPicker):
        def _onset_indices(self, data):
            indices = np.random.permutation(np.arange(len(data)))[:3]
            indices.sort()
            return indices

    def test_peak_picker_returns_variable_rate_time_series(self):
        data = ArrayWithUnits(
            np.zeros(100),
            dimensions=[TimeDimension(frequency=Seconds(1))])
        picker = BasePeakPickerTests.PeakPicker()
        results = picker._process(data).next()
        self.assertEqual(3, len(results))
        self.assertIsInstance(results, VariableRateTimeSeries)
