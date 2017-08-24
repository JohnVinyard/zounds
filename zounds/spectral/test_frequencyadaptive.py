import unittest2
from zounds.core import ArrayWithUnits
from zounds.timeseries import TimeDimension, Seconds, Milliseconds, SR11025
from zounds.spectral import \
    FrequencyBand, ExplicitFrequencyDimension, GeometricScale, ExplicitScale, \
    FrequencyDimension, LinearScale
from frequencyadaptive import FrequencyAdaptive
import numpy as np


class FrequencyAdaptiveTests(unittest2.TestCase):

    def test_raises_when_time_dimension_is_none(self):
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        self.assertRaises(
            ValueError, lambda: FrequencyAdaptive(arrs, None, scale))

    def test_raises_when_explicit_freq_dimension_and_non_contiguous_array(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa1 = FrequencyAdaptive(arrs, td, scale)
        arr = np.asarray(fa1)
        self.assertRaises(ValueError, lambda: FrequencyAdaptive(
            arr,
            td,
            scale=scale,
            explicit_freq_dimension=fa1.frequency_dimension))

    def test_can_construct_from_contiguous_array(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa1 = FrequencyAdaptive(arrs, td, scale)
        arr = np.asarray(fa1)
        fa2 = FrequencyAdaptive(
            arr, td, explicit_freq_dimension=fa1.frequency_dimension)
        self.assertEqual(fa1.shape, fa2.shape)
        self.assertEqual(fa1.scale, fa2.scale)
        self.assertEqual(fa1.time_dimension, fa2.time_dimension)
        self.assertEqual(fa1.frequency_dimension, fa2.frequency_dimension)

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
        arrs = [np.zeros((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        sliced = fa[:, scale[0]]
        self.assertEqual((10, 1), sliced.shape)

    def test_square_form_with_overlap(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.zeros((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        square = fa.square(50)

        self.assertEqual(3, square.ndim)
        self.assertEqual(10, square.shape[0])
        self.assertEqual(50, square.shape[1])
        self.assertEqual(120, square.shape[2])

        self.assertIsInstance(square, ArrayWithUnits)

        self.assertIsInstance(square.dimensions[0], TimeDimension)
        self.assertEqual(Milliseconds(5500), square.dimensions[0].end)
        self.assertEqual(Milliseconds(500), square.dimensions[0].frequency)
        self.assertEqual(Milliseconds(1000), square.dimensions[0].duration)

        self.assertIsInstance(square.dimensions[1], TimeDimension)

        self.assertIsInstance(square.dimensions[2], FrequencyDimension)
        self.assertEqual(scale, square.dimensions[2].scale)

    def test_square_form_with_overlap_do_overlap_add(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Milliseconds(500))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.zeros((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        square = fa.square(50, do_overlap_add=True)

        self.assertEqual(2, square.ndim)
        self.assertEqual(275, square.shape[0])
        self.assertEqual(120, square.shape[1])

        self.assertIsInstance(square, ArrayWithUnits)

        self.assertIsInstance(square.dimensions[0], TimeDimension)
        self.assertEqual(Milliseconds(5500), square.dimensions[0].end)
        self.assertEqual(Milliseconds(20), square.dimensions[0].frequency)
        self.assertEqual(Milliseconds(20), square.dimensions[0].duration)

        self.assertIsInstance(square.dimensions[1], FrequencyDimension)
        self.assertEqual(scale, square.dimensions[1].scale)

    def test_square_form_no_overlap(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.zeros((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        square = fa.square(50)

        self.assertEqual(3, square.ndim)

        self.assertEqual(10, square.shape[0])
        self.assertEqual(50, square.shape[1])
        self.assertEqual(120, square.shape[2])

        self.assertIsInstance(square, ArrayWithUnits)

        self.assertIsInstance(square.dimensions[0], TimeDimension)
        self.assertEqual(Seconds(10), square.dimensions[0].end)
        self.assertEqual(Milliseconds(1000), square.dimensions[0].frequency)
        self.assertEqual(Milliseconds(1000), square.dimensions[0].duration)

        self.assertIsInstance(square.dimensions[1], TimeDimension)

        self.assertIsInstance(square.dimensions[2], FrequencyDimension)
        self.assertEqual(scale, square.dimensions[2].scale)

    def test_square_form_no_overlap_do_overlap_add(self):
        td = TimeDimension(
            duration=Seconds(1),
            frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 120)
        arrs = [np.zeros((10, x)) for x in xrange(1, 121)]
        fa = FrequencyAdaptive(arrs, td, scale)
        square = fa.square(50, do_overlap_add=True)

        self.assertEqual(2, square.ndim)
        self.assertEqual(500, square.shape[0])
        self.assertEqual(120, square.shape[1])

        self.assertIsInstance(square, ArrayWithUnits)

        self.assertIsInstance(square.dimensions[0], TimeDimension)
        self.assertEqual(Seconds(10), square.dimensions[0].end)
        self.assertEqual(Milliseconds(20), square.dimensions[0].frequency)
        self.assertEqual(Milliseconds(20), square.dimensions[0].duration)

        self.assertIsInstance(square.dimensions[1], FrequencyDimension)
        self.assertEqual(scale, square.dimensions[1].scale)

    def test_from_arr_with_units(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)

        raw_arr = ArrayWithUnits(np.array(fa), fa.dimensions)

        fa2 = FrequencyAdaptive.from_array_with_units(raw_arr)
        self.assertIsInstance(fa2, FrequencyAdaptive)
        self.assertEqual(fa2.dimensions[0], fa.dimensions[0])
        self.assertEqual(fa2.dimensions[1], fa.dimensions[1])
        self.assertEqual(fa.shape, fa2.shape)
