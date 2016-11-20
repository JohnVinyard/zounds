import numpy as np
from zounds.nputil import sliding_window
from dimensions import IdentityDimension


class CustomSlice(object):
    def __init__(self):
        super(CustomSlice, self).__init__()


class ArrayWithUnits(np.ndarray):
    def __new__(cls, arr, dimensions):
        if arr.ndim != len(dimensions):
            raise ValueError('arr.ndim and len(dimensions) must match')
        obj = np.asarray(arr).view(cls)
        obj.dimensions = tuple(map(
                lambda x: IdentityDimension() if x is None else x, dimensions))
        for dim, size in zip(obj.dimensions, obj.shape):
            try:
                dim.size = size
            except AttributeError:
                pass
        return obj

    def concatenate(self, other):
        if self.dimensions == other.dimensions:
            return self.from_example(np.concatenate(self, other), self)
        else:
            raise ValueError('All dimensions must match to concatenate')

    @classmethod
    def concat(cls, arrs, axis=0):
        for arr in arrs[1:]:
            if arr.dimensions != arrs[0].dimensions:
                raise ValueError('All dimensions must match')
        return cls.from_example(np.concatenate(arrs, axis=axis), arrs[0])

    @classmethod
    def from_example(cls, data, example):
        return ArrayWithUnits(data, example.dimensions)

    def sum(self, axis=None, dtype=None, **kwargs):
        result = super(ArrayWithUnits, self).sum(axis, dtype, **kwargs)
        if axis is not None:
            new_dims = list(self.dimensions)
            new_dims.pop(axis)
            return self.__class__(result, new_dims)
        else:
            # we have a scalar
            return result

    def dot(self, b):
        result = super(ArrayWithUnits, self).dot(b)
        return self.__class__(result, self.dimensions[:result.ndim])

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
            new_dims = tuple(self._compute_new_dims(result, ws))
        except ValueError:
            new_dims = [IdentityDimension()] * result.ndim

        return self.__class__(result, new_dims)

    def _compute_new_dims(self, windowed, ws):
        for dimension, size, w in zip(self.dimensions, self.shape, ws):
            modified = dimension.modified_dimension(size, w)
            for m in modified:
                yield m

    def _compute_span(self, index):
        for sl in self._compute_indices(index):
            try:
                yield sl.stop - sl.start
            except AttributeError:
                yield sl

    def _compute_indices(self, index):
        dims_pos = 0
        for sl in index:
            if sl is None \
                    or isinstance(sl, int) \
                    or isinstance(sl, slice) \
                    or isinstance(sl, list):
                # burn one
                dims_pos += 1
                yield sl
            else:
                dim = self.dimensions[dims_pos]
                yield dim.integer_based_slice(sl)
                dims_pos += 1

    def _new_dims(self, index, new_arr):
        dims_pos = 0
        shape_pos = 0
        for sl in index:
            if sl is None:
                # additional dimension via np.newaxis
                yield IdentityDimension()
            elif isinstance(sl, int):
                # burn a dimension
                dims_pos += 1
                shape_pos += 1
            elif isinstance(sl, list):
                dims_pos += 1
                shape_pos += 1
                yield IdentityDimension()
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

    def __gt__(self, other):
        return np.asarray(self).__gt__(other)

    def __ge__(self, other):
        return np.asarray(self).__ge__(other)

    def __lt__(self, other):
        return np.asarray(self).__lt__(other)

    def __le__(self, other):
        return np.asarray(self).__le__(other)

    def __eq__(self, other):
        return np.asarray(self).__eq__(other)

    def _tuplify(self, a):
        if isinstance(a, list):
            t = set(map(lambda x: x.__class__, a))
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
        return self.__class__(arr, new_dims)
