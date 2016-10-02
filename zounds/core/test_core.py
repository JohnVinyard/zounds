import unittest2
import numpy as np
from axis import Dimension, ArrayWithUnits, CustomSlice, IdentityDimension


class ContrivedSlice(CustomSlice):
    def __init__(self, start=None, stop=None, step=None):
        super(ContrivedSlice, self).__init__()
        self.step = step
        self.stop = stop
        self.start = start


class ContrivedDimension(Dimension):
    def __init__(self, factor):
        super(ContrivedDimension, self).__init__()
        self.factor = factor

    def modified_dimension(self, size, windowsize):
        yield ContrivedDimension(self.factor * windowsize)
        yield ContrivedDimension(self.factor)

    def integer_based_slice(self, index):
        return slice(index.start // self.factor, index.stop // self.factor)

    def __eq__(self, other):
        try:
            return self.factor == other.factor
        except AttributeError:
            return False

    def __repr__(self):
        return 'ContrivedDimension(factor={factor})'.format(**self.__dict__)


class ContrivedDimension2(Dimension):
    def __init__(self, factor):
        super(ContrivedDimension2, self).__init__()


class ContrivedArray(ArrayWithUnits):
    def __new__(cls, arr, dimensions):
        return ArrayWithUnits.__new__(cls, arr, dimensions)


class CoreTests(unittest2.TestCase):
    def test_can_get_custom_slice(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (None, ContrivedDimension(10)))
        custom_slice = ContrivedSlice(50, 70)
        result = arr[5:7, custom_slice]
        self.assertEqual((2, 2), result.shape)

    def test_can_set_custom_slice(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (None, ContrivedDimension(10)))
        custom_slice = ContrivedSlice(50, 70)
        arr[5:7, custom_slice] = 1
        np.testing.assert_allclose(arr[5:7, 5:7], 1)

    def test_dimensions_must_match(self):
        raw = np.zeros((3, 3, 3))
        self.assertRaises(
                ValueError,
                lambda: ContrivedArray(raw, (None, ContrivedDimension(5))))

    def test_custom_dimensions_are_preserved_after_slice(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (None, ContrivedDimension(10)))
        custom_slice = ContrivedSlice(50, 70)
        result = arr[5:7, custom_slice]
        self.assertIsInstance(result.dimensions[0], IdentityDimension)
        self.assertEqual(ContrivedDimension(10), result.dimensions[1])

    def test_custom_dimensions_are_preserved_after_slice_that_removes_dimension(
            self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (None, ContrivedDimension(10)))
        custom_slice = ContrivedSlice(50, 70)
        result = arr[0, custom_slice]
        self.assertEqual(1, result.ndim)
        self.assertEqual(ContrivedDimension(10), result.dimensions[0])

    def test_can_get_single_value_from_custom_dimension(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        result = arr[0]
        self.assertEqual(0, result)

    def test_can_set_single_value_in_custom_dimension(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        arr[0] = 1
        self.assertEqual(1, arr[0])

    def test_custom_dimension_is_maintained_for_sliding_window_1d(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20),))
        self.assertIsInstance(new_arr, ContrivedArray)
        self.assertEqual((5, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[1], ContrivedDimension)
        self.assertEqual(20, new_arr.dimensions[0].factor)
        self.assertEqual(10, new_arr.dimensions[1].factor)

    def test_item_access_is_correct_after_sliding_window_1d(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20),))
        sliced = new_arr[ContrivedSlice(0, 40)]
        self.assertEqual((2, 2), sliced.shape)

    def test_custom_dimension_is_maintained_for_sw_1d_with_step(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        new_arr = arr.sliding_window(
                (ContrivedSlice(0, 20),), (ContrivedSlice(0, 10),))
        self.assertIsInstance(new_arr, ContrivedArray)
        self.assertEqual((9, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[1], ContrivedDimension)
        self.assertEqual(20, new_arr.dimensions[0].factor)
        self.assertEqual(10, new_arr.dimensions[1].factor)

    def test_custom_dimension_is_maintained_for_sliding_window(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (ContrivedDimension(10), None))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20), 10))
        self.assertIsInstance(new_arr, ContrivedArray)
        self.assertEqual((5, 2, 10), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[1], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[2], IdentityDimension)
        self.assertEqual(20, new_arr.dimensions[0].factor)
        self.assertEqual(10, new_arr.dimensions[1].factor)

    def test_custom_dimension_is_maintained_for_sw_with_step(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (ContrivedDimension(10), None))
        new_arr = arr.sliding_window(
                (ContrivedSlice(0, 20), 10), (ContrivedSlice(0, 10), 10))
        self.assertIsInstance(new_arr, ContrivedArray)
        self.assertEqual((9, 2, 10), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[1], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[2], IdentityDimension)
        self.assertEqual(20, new_arr.dimensions[0].factor)
        self.assertEqual(10, new_arr.dimensions[1].factor)

    def test_custom_dimension_is_not_maintained_for_sliding_window(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (ContrivedDimension(10), None))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20), 2))
        self.assertIsInstance(new_arr, ContrivedArray)
        self.assertEqual((25, 2, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[1], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[2], IdentityDimension)

    def test_custom_dimension_is_not_maintained_for_sw_with_step(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (ContrivedDimension(10), None))
        new_arr = arr.sliding_window(
                (ContrivedSlice(0, 20), 2), (ContrivedSlice(0, 10), 1))
        self.assertIsInstance(new_arr, ContrivedArray)
        self.assertEqual((81, 2, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[1], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[2], IdentityDimension)
