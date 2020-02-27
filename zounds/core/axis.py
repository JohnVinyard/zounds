import numpy as np
from zounds.nputil import sliding_window, windowed
from zounds.util import tuplify
from .dimensions import IdentityDimension


class CustomSlice(object):
    def __init__(self):
        super(CustomSlice, self).__init__()


class ArrayWithUnits(np.ndarray):
    """
    `ArrayWithUnits` is an :class:`numpy.ndarray` subclass that allows for
    indexing by more semantically meaningful slices.

    It supports most methods on :class:`numpy.ndarray`, and makes a best-effort
    to maintain meaningful dimensions throughout those operations.

    Args:
        arr (ndarray): The :class:`numpy.ndarray` instance containing the raw
            data for this instance
        dimensions (list or tuple): list or tuple of :class:`Dimension`-derived
            classes

    Raises:
        ValueError: when `arr.ndim` and `len(dimensions)` do not match

    Examples:
        >>> from zounds import ArrayWithUnits, TimeDimension, Seconds, TimeSlice
        >>> import numpy as np
        >>> data = np.zeros(100)
        >>> awu = ArrayWithUnits(data, [TimeDimension(Seconds(1))])
        >>> sliced = awu[TimeSlice(Seconds(10))]
        >>> sliced.shape
        (10,)

    See Also:
        :class:`IdentityDimension`
        :class:`~zounds.timeseries.TimeDimension`
        :class:`~zounds.spectral.FrequencyDimension`
    """

    def __new__(cls, arr, dimensions):
        if arr.ndim != len(dimensions):
            raise ValueError(
                'arr.ndim and len(dimensions) must match.  '
                'They were {arr.shape} and {dimensions}'.format(**locals()))

        obj = np.asarray(arr).view(cls)
        obj.dimensions = tuple([IdentityDimension() if d is None else d.copy() for d in dimensions])

        for dim, size in zip(obj.dimensions, obj.shape):
            try:
                dim.size = size
            except AttributeError:
                pass
            try:
                dim.validate(size)
            except AttributeError:
                pass

        return obj

    @property
    def T(self):
        arr = super(ArrayWithUnits, self).T
        return ArrayWithUnits(arr, self.dimensions[::-1])

    def kwargs(self):
        return self.__dict__

    def concatenate(self, other):
        if self.dimensions == other.dimensions:
            return self.from_example(np.concatenate([self, other]), self)
        else:
            raise ValueError('All dimensions must match to concatenate')

    def reshape(self, shape, order='C'):
        non_one = lambda x: abs(x) != 1

        if tuple(shape) == tuple(filter(non_one, self.shape)):
            # the new shape is this array's shape will all ones removed
            return self.squeeze()
        elif tuple(self.shape) == tuple(filter(non_one, shape)):
            # the new shape just adds some single dimension axes
            dims = iter(self.dimensions)
            new_dims = [
                IdentityDimension() if abs(size) == 1 else next(dims)
                for size in shape
            ]
            raw = np.asarray(self)
            return ArrayWithUnits(raw.reshape(shape), new_dims)

        # treat every axis as an IdentityDimension
        raw = np.asarray(self)
        return ArrayWithUnits(
            np.reshape(raw, shape, order),
            [IdentityDimension() for _ in shape])

    def squeeze(self):
        zipped = [x for x in zip(self.shape, self.dimensions) if x[0] > 1]
        return ArrayWithUnits(
            super().reshape([s for s, _ in zipped]),
            [d for _, d in zipped])

    @classmethod
    def concat(cls, arrs, axis=0):
        for arr in arrs[1:]:
            if arr.dimensions != arrs[0].dimensions:
                raise ValueError('All dimensions must match')
        return cls.from_example(np.concatenate(arrs, axis=axis), arrs[0])

    @classmethod
    def from_example(cls, data, example):
        """
        Produce a new :class:`ArrayWithUnits` instance given some raw data and
        an example instance that has the desired dimensions
        """
        return ArrayWithUnits(data, example.dimensions)

    @classmethod
    def zeros(cls, example):
        return cls.from_example(
            np.zeros(example.shape, dtype=example.dtype), example)

    def zeros_like(self):
        return self.__class__.zeros(self)

    def _apply_reduction_to_dimensions(self, result, axis, keepdims):
        if axis is None:
            return result

        ndims = len(self.dimensions)
        reduced_axes = set([ndims + a if a < 0 else a for a in tuplify(axis)])
        all_axes = set(range(ndims))

        if keepdims:
            new_dims = [
                IdentityDimension() if i in reduced_axes else dim
                for i, dim in enumerate(self.dimensions)]
        else:
            remaining_axes = sorted(all_axes - reduced_axes)
            new_dims = [self.dimensions[i] for i in remaining_axes]

        return ArrayWithUnits(result, new_dims)

    def sum(self, axis=None, dtype=None, keepdims=False, **kwargs):
        result = super(ArrayWithUnits, self).sum(
            axis, dtype, keepdims=keepdims, **kwargs)
        return self._apply_reduction_to_dimensions(result, axis, keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        result = super(ArrayWithUnits, self).max(
            axis=axis, out=out, keepdims=keepdims)
        return self._apply_reduction_to_dimensions(result, axis, keepdims)

    def dot(self, b):
        result = super(ArrayWithUnits, self).dot(b)
        return self.__class__(result, self.dimensions[:result.ndim])

    def packbits(self, axis=None):
        arr = np.packbits(self, axis=axis)
        return ArrayWithUnits(arr, self.dimensions)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dimensions = getattr(obj, 'dimensions', None)

    def __array_wrap__(self, obj, context=None):
        if len(self.dimensions) != obj.ndim:
            if obj.ndim == 0:
                return obj[0]
            return np.asarray(obj)
        return np.ndarray.__array_wrap__(self, obj, context)

    def sliding_window(self, windowsize, stepsize=None):
        ws = tuple(self._compute_span(windowsize))
        ss = tuple(self._compute_span(stepsize)) if stepsize else ws
        result = sliding_window(self, ws, ss)

        try:
            new_dims = tuple(self._compute_new_dims(result, ws, ss))
        except ValueError:
            new_dims = [IdentityDimension()] * result.ndim

        return ArrayWithUnits(result, new_dims)

    def _sliding_window_integer_slices(self, windowsize, stepsize=None):
        windowsize = [windowsize] + [slice(None) for _ in self.dimensions[1:]]
        ws = tuple(self._compute_span(windowsize))
        if stepsize:
            stepsize = [stepsize] + [slice(None) for _ in self.dimensions[1:]]
            ss = tuple(self._compute_span(stepsize))
        else:
            ss = ws

        return ws, ss

    def sliding_window_with_leftovers(
            self, windowsize, stepsize=None, dopad=False):

        ws, ss = \
            self._sliding_window_integer_slices(windowsize, stepsize)

        leftovers, result = windowed(self, ws[0], ss[0], dopad)

        if not result.size:
            return self, ArrayWithUnits(result, self.dimensions)

        try:
            new_dims = tuple(self._compute_new_dims(result, ws, ss))
        except ValueError:
            new_dims = [IdentityDimension()] * result.ndim

        return leftovers, ArrayWithUnits(result, new_dims)

    def _compute_new_dims(self, windowed, ws, ss):
        for dimension, size, w, s in zip(self.dimensions, self.shape, ws, ss):
            try:
                modified = dimension.modified_dimension(size, w, s)
                for m in modified:
                    yield m
            except NotImplementedError:
                yield dimension

    def _compute_span(self, index):
        for sl in self._compute_indices(index):
            try:
                yield sl.stop - sl.start
            except (AttributeError, TypeError):
                yield sl

    def _is_integer_based_slice(self, sl):
        if not isinstance(sl, slice):
            return False

        try:
            return \
                (sl.start is None or sl.start.bit_length()) \
                and (sl.stop is None or sl.stop.bit_length())
        except AttributeError:
            return False

    def _compute_indices(self, index):
        dims_pos = 0
        for sl in index:
            if sl is None \
                    or isinstance(sl, int) \
                    or self._is_integer_based_slice(sl) \
                    or isinstance(sl, list):
                # burn one
                dims_pos += 1
                yield sl
            elif sl is Ellipsis:
                dims_pos += len(self.dimensions) - (len(index) - 1)
                yield Ellipsis
            else:
                dim = self.dimensions[dims_pos]
                yield dim.integer_based_slice(sl)
                dims_pos += 1

    def _new_dims(self, index, new_arr):
        dims_pos = 0
        shape_pos = 0
        not_ellipsis_or_none = len([x for x in index if x is not Ellipsis and x is not None])
        for sl in index:
            if sl is None:
                # additional dimension via np.newaxis
                yield IdentityDimension()
            elif isinstance(sl, int):
                # burn a dimension
                dims_pos += 1
            elif isinstance(sl, list):
                dims_pos += 1
                shape_pos += 1
                yield IdentityDimension()
            elif sl is Ellipsis:
                ellipsis_size = len(self.dimensions) - not_ellipsis_or_none
                for i in range(ellipsis_size):
                    yield self.dimensions[dims_pos]
                    dims_pos += 1
                    shape_pos += 1
            else:
                try:
                    dim = self.dimensions[dims_pos]
                    shape = new_arr.shape[shape_pos]
                    yield dim.metaslice(sl, shape)
                except IndexError:
                    yield dim
                dims_pos += 1
                shape_pos += 1

        # Return any leftover dimensions
        for dim in self.dimensions[dims_pos:]:
            yield dim

    def _tuplify(self, a):
        if isinstance(a, list):
            t = set([x.__class__ for x in a])
            if len(t) > 1:
                raise ValueError('a must be homogeneous')
            t = list(t)[0]
            if t == slice:
                return a
            else:
                return a,
        if isinstance(a, np.ndarray) and a.dtype == np.bool:
            return a,
        try:
            return tuple(a)
        except TypeError:
            return a,

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __setitem__(self, index, value):
        index = self._tuplify(index)
        indices = tuple(self._compute_indices(index))
        super(ArrayWithUnits, self).__setitem__(indices, value)

    def __getitem__(self, index):
        if isinstance(index, np.ndarray) \
                and index.dtype == np.bool:
            return np.asarray(self)[index]

        if self.ndim == 1 and isinstance(index, int):
            return np.asarray(self)[index]

        index = self._tuplify(index)
        indices = tuple(self._compute_indices(index))
        arr = super(ArrayWithUnits, self).__getitem__(indices)
        new_dims = tuple(self._new_dims(index, arr))
        return ArrayWithUnits(arr, new_dims)
