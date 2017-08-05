import unittest2
from tfrepresentation import FrequencyDimension, ExplicitFrequencyDimension
from frequencyscale import FrequencyBand, LinearScale, LogScale, GeometricScale


class FrequencyDimensionTests(unittest2.TestCase):
    def test_equal(self):
        fd1 = FrequencyDimension(LinearScale(FrequencyBand(20, 10000), 100))
        fd2 = FrequencyDimension(LinearScale(FrequencyBand(20, 10000), 100))
        self.assertEqual(fd1, fd2)

    def test_not_equal(self):
        fd1 = FrequencyDimension(LinearScale(FrequencyBand(20, 10000), 100))
        fd2 = FrequencyDimension(LogScale(FrequencyBand(20, 10000), 100))
        self.assertNotEqual(fd1, fd2)


class ExplicitFrequencyDimensionTests(unittest2.TestCase):
    def test_equal(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)

        scale2 = GeometricScale(20, 5000, 0.02, 3)
        slices2 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim2 = ExplicitFrequencyDimension(scale2, slices2)

        self.assertEqual(dim1, dim2)

    def test_not_equal_due_to_slices(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)

        scale2 = GeometricScale(20, 5000, 0.02, 3)
        slices2 = [slice(0, 10), slice(10, 100), slice(100, 1001)]
        dim2 = ExplicitFrequencyDimension(scale2, slices2)

        self.assertNotEqual(dim1, dim2)

    def test_not_equal_due_to_scales(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)

        scale2 = GeometricScale(20, 5000, 0.02, 3)
        slices2 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim2 = ExplicitFrequencyDimension(scale2, slices2)

        self.assertNotEqual(dim1, dim2)

    def test_raises_when_scale_and_slices_are_different_sizes(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        slices1 = [slice(0, 10), slice(10, 100)]
        self.assertRaises(
            ValueError, lambda: ExplicitFrequencyDimension(scale1, slices1))

    def test_metaslice(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        bands = list(scale1)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)
        dim2 = dim1.metaslice(FrequencyBand(15, 1000), 2)
        self.assertEqual(bands[:2], list(dim2.scale)[:2])
        self.assertEqual(slices1[:2], dim2.slices[:2])

    def test_metaslice_exact_matching_band(self):
        scale = GeometricScale(20, 5000, 0.05, 120)
        # the values of the slices don't matter for this test
        slices = [slice(0, 10) for _ in xrange(len(scale))]
        dim = ExplicitFrequencyDimension(scale, slices)
        dim2 = dim.metaslice(scale[0], 1)
        self.assertEqual(1, len(dim2.scale))
        self.assertEqual(1, len(dim2.slices))
        self.assertEqual(dim.scale[0], dim2.scale[0])
        self.assertEqual(dim.slices[0], dim2.slices[0])

    def test_metaslice_fuzzy_matching_band(self):
        scale = GeometricScale(20, 5000, 0.05, 120)
        # the values of the slices don't matter for this test
        slices = [slice(0, 10) for _ in xrange(len(scale))]
        dim = ExplicitFrequencyDimension(scale, slices)
        first_band = scale[0]
        band = FrequencyBand(first_band.start_hz, first_band.stop_hz + 1)
        dim2 = dim.metaslice(band, 3)
        self.assertEqual(3, len(dim2.scale))
        self.assertEqual(3, len(dim2.slices))
        self.assertEqual(dim.scale[0], dim2.scale[0])
        self.assertEqual(dim.scale[1], dim2.scale[1])
        self.assertEqual(dim.scale[2], dim2.scale[2])
        self.assertEqual(dim.slices[0], dim2.slices[0])
        self.assertEqual(dim.slices[1], dim2.slices[1])
        self.assertEqual(dim.slices[2], dim2.slices[2])

    def test_can_get_slice_when_perfectly_corresponds_to_band(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        bands = list(scale1)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)
        self.assertEqual(slices1[1], dim1.integer_based_slice(bands[1]))

    def test_can_get_slice_with_overlap(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        bands = list(scale1)
        slices1 = [slice(0, 10), slice(5, 100), slice(50, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)
        self.assertEqual(slices1[1], dim1.integer_based_slice(bands[1]))

    def test_is_valid_when_size_corresponds_to_last_slice_end(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)
        self.assertTrue(dim1.validate(1000))

    def test_is_not_valid_when_size_does_not_correspond_to_last_slice(self):
        scale1 = GeometricScale(20, 5000, 0.02, 3)
        slices1 = [slice(0, 10), slice(10, 100), slice(100, 1000)]
        dim1 = ExplicitFrequencyDimension(scale1, slices1)
        self.assertFalse(dim1.validate(2000))
