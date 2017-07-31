import unittest2
from zounds.core import ArrayWithUnits
from zounds.timeseries import TimeDimension, Seconds, Milliseconds
from zounds.spectral import \
    FrequencyBand, ExplicitFrequencyDimension, GeometricScale, ExplicitScale
from frequencyadaptive import FrequencyAdaptive
import numpy as np


class FrequencyAdaptiveTests(unittest2.TestCase):
    def test_can_construct_instance(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)
        self.assertEqual((10, 55), fa.shape)
        self.assertIsInstance(fa.dimensions[0], TimeDimension)
        self.assertIsInstance(fa.dimensions[1], ExplicitFrequencyDimension)

    def test_can_concatenate_instances(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)

        arrs2 = [np.zeros((20, x)) for x in xrange(1, 11)]
        fa2 = FrequencyAdaptive(arrs2, td, scale)

        result = fa.concatenate(fa2)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((30, 55), result.shape)

    def test_can_get_single_frequency_band_over_entire_duration(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)

        single_band = fa[:, scale[5]]

        self.assertIsInstance(single_band, ArrayWithUnits)
        self.assertIsInstance(single_band.dimensions[0], TimeDimension)
        self.assertIsInstance(
            single_band.dimensions[1], ExplicitFrequencyDimension)
        self.assertEqual(1, len(single_band.dimensions[1].scale))
        self.assertEqual(1, len(single_band.dimensions[1].slices))

    def test_can_apply_frequency_slice_across_multiple_bands(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)
        band = FrequencyBand(300, 3030)

        fa2 = fa[:, band]

        self.assertIsInstance(fa2, ArrayWithUnits)
        self.assertEqual(td, fa2.dimensions[0])
        self.assertIsInstance(fa2.dimensions[1], ExplicitFrequencyDimension)
        self.assertIsInstance(fa2.dimensions[1].scale, ExplicitScale)

    def test_can_assign_to_multi_band_frequency_slice(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)
        band = FrequencyBand(300, 3030)
        fa[:, band] = 1
        int_slice = fa.dimensions[1].integer_based_slice(band)
        np.testing.assert_allclose(fa[:, int_slice], 1)

    def test_can_access_single_frequency_band(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        print 'SCALE SLICE BEFORE DECODE', scale.get_slice(scale[0])
        arrs = [np.zeros((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        sliced = fa[:, scale[0]]
        self.assertEqual((10, 1), sliced.shape)

    def test_square_form(self):
        self.fail()

    def test_from_arr_with_units(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)
        print fa.dimensions[1].slices

        raw_arr = ArrayWithUnits(np.array(fa), fa.dimensions)

        fa2 = FrequencyAdaptive.from_array_with_units(raw_arr)
        self.assertIsInstance(fa2, FrequencyAdaptive)
        self.assertEqual(fa2.dimensions[0], fa.dimensions[0])
        self.assertEqual(fa2.dimensions[1], fa.dimensions[1])
        self.assertEqual(fa.shape, fa2.shape)
