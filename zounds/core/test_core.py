import unittest2
import numpy as np
from dimensions import Dimension, IdentityDimension
from axis import ArrayWithUnits, CustomSlice
from string import ascii_lowercase


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
        self.size = None

    def copy(self):
        return ContrivedDimension(self.factor)

    def modified_dimension(self, size, windowsize, stepsize=None):
        yield ContrivedDimension(self.factor * windowsize)
        yield ContrivedDimension(self.factor)

    def integer_based_slice(self, index):
        if isinstance(index, ContrivedSlice):
            return slice(index.start // self.factor, index.stop // self.factor)
        else:
            return index

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
        self.factor = factor
        self.size = None

    def copy(self):
        return ContrivedDimension2(self.factor)

    def modified_dimension(self, size, windowsize, stepsize=None):
        yield ContrivedDimension(self.factor * windowsize)
        yield ContrivedDimension(self.factor)

    def integer_based_slice(self, index):
        if isinstance(index, ContrivedSlice):
            return slice(index.start // self.factor, index.stop // self.factor)
        else:
            return index

    def __eq__(self, other):
        try:
            return self.factor == other.factor
        except AttributeError:
            return False

    def __repr__(self):
        return 'ContrivedDimension2(factor={factor})'.format(**self.__dict__)


class AsciiCharacterDimension(Dimension):
    def __init__(self, labels):
        super(AsciiCharacterDimension, self).__init__()
        self.labels = labels

    def copy(self):
        return AsciiCharacterDimension(self.labels)

    def modified_dimension(self, size, windowsize, stepsize=None):
        raise NotImplementedError()

    def metaslice(self, index, size):
        return AsciiCharacterDimension(self.labels[index])

    def integer_based_slice(self, index):
        return index


class ContrivedArray(ArrayWithUnits):
    def __new__(cls, arr, dimensions):
        return ArrayWithUnits.__new__(cls, arr, dimensions)


class CoreTests(unittest2.TestCase):
    def test_maintain_array_with_units_with_boolean_condition(self):
        arr = ArrayWithUnits(
            np.random.random_sample((10, 100)) - 0.5,
            [ContrivedDimension(10), ContrivedDimension2(10)])
        binary = arr >= 0
        self.assertIsInstance(binary, ArrayWithUnits)
        self.assertIsInstance(arr.dimensions[0], ContrivedDimension)
        self.assertIsInstance(arr.dimensions[1], ContrivedDimension2)

    def test_can_reshape_and_downgrade_to_identity_dimension(self):
        arr = ArrayWithUnits(
            np.zeros((100, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        flattened = arr.reshape((-1,))
        self.assertEqual((1000,), flattened.shape)
        self.assertIsInstance(flattened, ArrayWithUnits)
        self.assertIsInstance(flattened.dimensions[0], IdentityDimension)

    def test_can_maintain_array_with_units_when_reshaping_2d(self):
        arr = ArrayWithUnits(
            np.zeros((1, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        squeezed = arr.reshape((10,))
        self.assertEqual((10,), squeezed.shape)
        self.assertIsInstance(squeezed, ArrayWithUnits)
        self.assertIsInstance(squeezed.dimensions[0], ContrivedDimension2)

    def test_can_maintain_array_with_units_when_squeezing_2d(self):
        arr = ArrayWithUnits(
            np.zeros((1, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        squeezed = arr.squeeze()
        self.assertEqual((10,), squeezed.shape)
        self.assertIsInstance(squeezed, ArrayWithUnits)
        self.assertIsInstance(squeezed.dimensions[0], ContrivedDimension2)

    def test_can_maintain_array_with_units_when_reshaping_3d(self):
        arr = ArrayWithUnits(
            np.zeros((3, 1, 10)),
            [
                ContrivedDimension(10),
                IdentityDimension(),
                ContrivedDimension2(10)
            ])
        squeezed = arr.reshape((3, 10))
        self.assertEqual((3, 10), squeezed.shape)
        self.assertIsInstance(squeezed, ArrayWithUnits)
        self.assertIsInstance(squeezed.dimensions[0], ContrivedDimension)
        self.assertIsInstance(squeezed.dimensions[1], ContrivedDimension2)

    def test_can_maintain_array_with_units_when_squeezing_3d(self):
        arr = ArrayWithUnits(
            np.zeros((3, 1, 10)),
            [
                ContrivedDimension(10),
                IdentityDimension(),
                ContrivedDimension2(10)
            ])
        squeezed = arr.squeeze()
        self.assertEqual((3, 10), squeezed.shape)
        self.assertIsInstance(squeezed, ArrayWithUnits)
        self.assertIsInstance(squeezed.dimensions[0], ContrivedDimension)
        self.assertIsInstance(squeezed.dimensions[1], ContrivedDimension2)

    def test_squeeze_with_no_single_dimension_is_unchanged(self):
        arr = ArrayWithUnits(
            np.zeros((3, 2, 10)),
            [
                ContrivedDimension(10),
                IdentityDimension(),
                ContrivedDimension2(10)
            ])
        squeezed = arr.squeeze()
        self.assertEqual((3, 2, 10), squeezed.shape)
        self.assertIsInstance(squeezed, ArrayWithUnits)
        self.assertIsInstance(squeezed.dimensions[0], ContrivedDimension)
        self.assertIsInstance(squeezed.dimensions[1], IdentityDimension)
        self.assertIsInstance(squeezed.dimensions[2], ContrivedDimension2)

    def test_can_add_dimension_of_size_one_to_2d_array(self):
        arr = ArrayWithUnits(
            np.zeros((3, 10)),
            [
                ContrivedDimension(10),
                ContrivedDimension2(10)
            ])
        rs = arr.reshape((3, 1, 10))
        self.assertIsInstance(rs, ArrayWithUnits)
        self.assertIsInstance(rs.dimensions[0], ContrivedDimension)
        self.assertIsInstance(rs.dimensions[1], IdentityDimension)
        self.assertIsInstance(rs.dimensions[2], ContrivedDimension2)

    def test_can_add_dimension_of_size_one_to_1d_array(self):
        arr = ArrayWithUnits(np.zeros((3,)), [ContrivedDimension(10)])
        rs = arr.reshape((1, 1, arr.size))
        self.assertIsInstance(rs, ArrayWithUnits)
        self.assertIsInstance(rs.dimensions[0], IdentityDimension)
        self.assertIsInstance(rs.dimensions[1], IdentityDimension)
        self.assertIsInstance(rs.dimensions[2], ContrivedDimension)

    def test_can_add_dimension_of_size_one_with_wildcard_axis(self):
        arr = ArrayWithUnits(np.zeros((3,)), [ContrivedDimension(10)])
        rs = arr.reshape((-1, 1, arr.size))
        self.assertIsInstance(rs, ArrayWithUnits)
        print rs.dimensions
        self.assertIsInstance(rs.dimensions[0], IdentityDimension)
        self.assertIsInstance(rs.dimensions[1], IdentityDimension)
        self.assertIsInstance(rs.dimensions[2], ContrivedDimension)

    def test_identity_dimension_when_existing_dim_split_across_axes(self):
        arr = ArrayWithUnits(np.zeros((12,)), [ContrivedDimension(10)])
        rs = arr.reshape((2, 2, 3))
        self.assertIsInstance(rs, ArrayWithUnits)
        self.assertIsInstance(rs.dimensions[0], IdentityDimension)
        self.assertIsInstance(rs.dimensions[1], IdentityDimension)
        self.assertIsInstance(rs.dimensions[2], IdentityDimension)

    def test_identity_dimension_when_existing_dim_split_with_wildcard(self):
        arr = ArrayWithUnits(np.zeros((12,)), [ContrivedDimension(10)])
        rs = arr.reshape((2, -1, 3))
        self.assertIsInstance(rs, ArrayWithUnits)
        self.assertIsInstance(rs.dimensions[0], IdentityDimension)
        self.assertIsInstance(rs.dimensions[1], IdentityDimension)
        self.assertIsInstance(rs.dimensions[2], IdentityDimension)

    def test_zeros_like(self):
        arr = ArrayWithUnits(
            np.zeros((3, 2, 10)),
            [
                ContrivedDimension(10),
                IdentityDimension(),
                ContrivedDimension2(10)
            ])
        zeros = arr.zeros_like()
        self.assertEqual(zeros.shape, arr.shape)
        self.assertEqual(zeros.dimensions, arr.dimensions)
        np.testing.assert_allclose(0, zeros)

    def test_assigns_size_where_appropriate(self):
        arr = ArrayWithUnits(
            np.zeros((100, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        self.assertEqual(10, arr.dimensions[1].size)

    def test_can_create_new_array_from_example(self):
        arr = ArrayWithUnits(
            np.zeros((100, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        arr2 = ArrayWithUnits.from_example(np.zeros((90, 5)), arr)
        self.assertSequenceEqual(arr.dimensions, arr2.dimensions)
        self.assertEqual((90, 5), arr2.shape)

    def test_zeros(self):
        arr = ArrayWithUnits(
            np.zeros((100, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        arr2 = ArrayWithUnits.zeros(arr)
        self.assertEqual(arr.shape, arr2.shape)
        self.assertSequenceEqual(arr.dimensions, arr2.dimensions)
        np.testing.assert_allclose(arr2, 0)

    def test_zeros_dtype(self):
        arr = ArrayWithUnits(
            np.zeros((100, 10), dtype=np.complex128),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        arr2 = ArrayWithUnits.zeros(arr)
        self.assertEqual(arr.dtype, arr2.dtype)

    def test_size_is_not_modified_on_example_dimensions(self):
        arr = ArrayWithUnits(
            np.zeros((100, 10)),
            [ContrivedDimension(10), ContrivedDimension2(10)])
        arr2 = ArrayWithUnits.from_example(np.zeros((90, 5)), arr)
        self.assertEqual(100, arr.dimensions[0].size)
        self.assertEqual(10, arr.dimensions[1].size)
        self.assertEqual(90, arr2.dimensions[0].size)
        self.assertEqual(5, arr2.dimensions[1].size)

    def test_can_use_ellipsis_to_get_entire_array(self):
        raw = np.zeros((10, 10, 10))
        dims = (
            ContrivedDimension(10),
            ContrivedDimension(10),
            ContrivedDimension(10)
        )
        arr = ContrivedArray(raw, dims)
        result = arr[...]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((10, 10, 10), result.shape)
        self.assertEqual(3, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension)
        self.assertIsInstance(result.dimensions[2], ContrivedDimension)

    def test_can_use_ellipsis_to_get_last_two_dimensions(self):
        raw = np.zeros((10, 10, 10))
        dims = (
            ContrivedDimension(10),
            ContrivedDimension(10),
            ContrivedDimension(10)
        )
        arr = ContrivedArray(raw, dims)
        result = arr[ContrivedSlice(10, 30), ...]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((2, 10, 10), result.shape)
        self.assertEqual(3, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension)
        self.assertIsInstance(result.dimensions[2], ContrivedDimension)

    def test_can_use_ellipsis_to_get_first_two_dimensions(self):
        raw = np.zeros((10, 10, 10))
        dims = (
            ContrivedDimension(10),
            ContrivedDimension(10),
            ContrivedDimension(10)
        )
        arr = ContrivedArray(raw, dims)
        result = arr[..., ContrivedSlice(10, 30)]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((10, 10, 2), result.shape)
        self.assertEqual(3, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension)
        self.assertIsInstance(result.dimensions[2], ContrivedDimension)

    def test_can_multiply(self):
        raw = np.ones((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr * 10
        np.testing.assert_allclose(result, 10)

    def test_can_get_single_scalar_from_max_with_no_axis(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.max()
        self.assertIsInstance(result, float)
        self.assertEqual(0, result)

    def test_max_array_maintains_correct_dimensions_axis_0(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.max(axis=0)
        self.assertEqual((9,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension2)

    def test_max_array_maintains_correct_dimensions_axis_1(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.max(axis=1)
        self.assertEqual((8,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)

    def test_max_supports_keepdims(self):
        raw = np.random.random_sample((10, 9, 8, 7))
        arr = ArrayWithUnits(
            raw,
            dimensions=[
                ContrivedDimension(10),
                ContrivedDimension2(10),
                ContrivedDimension(10),
                ContrivedDimension2(10)
            ])
        result = arr.max(axis=(1, 2), keepdims=True)
        self.assertEqual((10, 1, 1, 7), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(4, len(result.dimensions))
        print result.dimensions
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], IdentityDimension)
        self.assertIsInstance(result.dimensions[2], IdentityDimension)
        self.assertIsInstance(result.dimensions[3], ContrivedDimension2)

    def test_max_supports_multiple_axes(self):
        raw = np.random.random_sample((10, 9, 8, 7))
        arr = ArrayWithUnits(
            raw,
            dimensions=[
                ContrivedDimension(10),
                IdentityDimension(),
                IdentityDimension(),
                ContrivedDimension2(10)
            ])
        result = arr.max(axis=(1, 2))
        self.assertEqual((10, 7), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(2, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension2)

    def test_get_single_scalar_from_sum_with_no_axis(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.sum()
        self.assertIsInstance(result, float)
        self.assertEqual(0, result)

    def test_array_maintains_correct_dimension_after_reduction(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.sum(axis=1)
        self.assertEqual((8,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)

    def test_array_maintains_correct_dimension_after_reduction2(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.sum(axis=0)
        self.assertEqual((9,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension2)

    def test_array_maintains_correct_dimension_after_reduction3(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = np.sum(arr, axis=0)
        self.assertEqual((9,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension2)

    def test_array_cannot_maintain_correct_dimension(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.sum(axis=1)
        self.assertEqual((10,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)

    def test_array_maintains_correct_dimensions_after_dot(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr.dot(np.zeros(9))
        self.assertEqual((8,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)

    @unittest2.skip(
        'this test fails because there is no hook to intercept this call')
    def test_array_maintains_correct_dimensions_after_dot2(self):
        raw = np.zeros((8, 9))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = np.dot(arr, np.zeros(9))
        self.assertEqual((8,), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)

    def test_can_select_subset_using_boolean_array(self):
        raw = np.arange(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        result = arr[arr >= 5]
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual((5,), result.shape)

    def test_correct_result_of_indexing_using_boolean_array(self):
        raw = np.random.random_sample((8, 9))
        arr = ContrivedArray(raw, (ContrivedDimension(10), IdentityDimension()))
        result = arr[arr > 0.5]
        self.assertIsInstance(result, np.ndarray)
        self.assertNotIsInstance(result, ArrayWithUnits)

    def test_can_select_subset_using_equality(self):
        raw = np.arange(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        result = arr[(arr == 5) | (arr == 6)]
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual((2,), result.shape)

    def test_can_set_subset_using_boolean_array(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        arr[arr == 0] = 10
        np.testing.assert_allclose(arr, 10)

    def test_can_apply_new_axis(self):
        raw = np.zeros((3, 5))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[None, -1]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((1, 5), result.shape)
        self.assertEqual(2, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], IdentityDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension2)

    def test_can_apply_new_axis_at_end(self):
        raw = np.zeros((3, 5))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[-1, None]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((1, 5), result.shape)
        self.assertEqual(2, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], IdentityDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension2)

    def test_can_apply_new_axis_in_middle(self):
        raw = np.zeros((3, 5))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[:, None, :]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((3, 1, 5), result.shape)
        self.assertEqual(3, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], IdentityDimension)
        self.assertIsInstance(result.dimensions[2], ContrivedDimension2)

    def test_can_apply_two_new_axes(self):
        raw = np.zeros((3, 5))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[None, None, -1]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((1, 1, 5), result.shape)
        self.assertEqual(3, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], IdentityDimension)
        self.assertIsInstance(result.dimensions[1], IdentityDimension)
        self.assertIsInstance(result.dimensions[2], ContrivedDimension2)

    def test_correct_axis_is_preserved(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[0]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(1, len(result.dimensions))
        self.assertEqual((10,), result.shape)
        self.assertIsInstance(result.dimensions[0], ContrivedDimension2)

    def test_multiple_axis_types(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[ContrivedSlice(20, 40), ContrivedSlice(20, 50)]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(2, len(result.dimensions))
        self.assertEqual((2, 3), result.shape)
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension2)

    def test_can_use_list_of_integers_as_index(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[[1, 3, 5]]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((3, 10), result.shape)
        self.assertEqual(2, len(result.dimensions))
        self.assertIsInstance(result.dimensions[0], IdentityDimension)
        self.assertIsInstance(result.dimensions[1], ContrivedDimension2)

    def test_can_use_multiple_integers(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        result = arr[0, 0]
        self.assertEqual(0, result)

    def test_1d(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        result = arr[1:4]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((3,), result.shape)
        self.assertIsInstance(result.dimensions[0], ContrivedDimension)

    def test_can_get_modified_dimension(self):
        raw = np.zeros(10)
        arr = ContrivedArray(
            raw, (AsciiCharacterDimension(ascii_lowercase[:10]),))
        result = arr[1:4]
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual((3,), result.shape)
        self.assertIsInstance(result.dimensions[0], AsciiCharacterDimension)
        self.assertEqual(ascii_lowercase[1:4], result.dimensions[0].labels)

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

    def test_can_set_from_list_of_integers(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(
            raw, (ContrivedDimension(10), ContrivedDimension2(10)))
        arr[[1, 3, 5]] = 5
        np.testing.assert_allclose(arr[[1, 3, 5]], 5)

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
        self.assertEqual((2,), result.shape)
        self.assertEqual(ContrivedDimension(10), result.dimensions[0])

    def test_can_get_single_value_from_custom_dimension(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        result = arr[0]
        self.assertEqual(0, result)
        self.assertIsInstance(result, float)

    def test_can_set_single_value_in_custom_dimension(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        arr[0] = 1
        self.assertEqual(1, arr[0])

    def test_sliding_window_maintains_dtype(self):
        raw = np.zeros(10, dtype=np.uint8)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20),))
        self.assertEqual(np.uint8, new_arr.dtype)

    def test_custom_dimension_is_maintained_for_sliding_window_1d(self):
        raw = np.zeros(10)
        arr = ContrivedArray(raw, (ContrivedDimension(10),))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20),))
        self.assertIsInstance(new_arr, ArrayWithUnits)
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
        self.assertIsInstance(new_arr, ArrayWithUnits)
        self.assertEqual((9, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], ContrivedDimension)
        self.assertIsInstance(new_arr.dimensions[1], ContrivedDimension)
        self.assertEqual(20, new_arr.dimensions[0].factor)
        self.assertEqual(10, new_arr.dimensions[1].factor)

    def test_custom_dimension_is_maintained_for_sliding_window(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (ContrivedDimension(10), None))
        new_arr = arr.sliding_window((ContrivedSlice(0, 20), 10))
        self.assertIsInstance(new_arr, ArrayWithUnits)
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
        self.assertIsInstance(new_arr, ArrayWithUnits)
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
        self.assertIsInstance(new_arr, ArrayWithUnits)
        self.assertEqual((25, 2, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[1], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[2], IdentityDimension)

    def test_custom_dimension_is_not_maintained_for_sw_with_step(self):
        raw = np.zeros((10, 10))
        arr = ContrivedArray(raw, (ContrivedDimension(10), None))
        new_arr = arr.sliding_window(
            (ContrivedSlice(0, 20), 2), (ContrivedSlice(0, 10), 1))
        self.assertIsInstance(new_arr, ArrayWithUnits)
        self.assertEqual((81, 2, 2), new_arr.shape)
        self.assertIsInstance(new_arr.dimensions[0], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[1], IdentityDimension)
        self.assertIsInstance(new_arr.dimensions[2], IdentityDimension)
