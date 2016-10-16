import numpy as np
from zounds.nputil import sliding_window


class Dimension(object):
    """
    Class representing one dimension of a numpy array, making custom slices
    (e.g., time spans or frequency bands) possible
    """

    def __init__(self):
        super(Dimension, self).__init__()

    def modified_dimension(self, size, windowsize):
        raise NotImplementedError()

    def metaslice(self, index, size):
        return self

    def integer_based_slice(self, index):
        raise NotImplementedError()


class IdentityDimension(Dimension):
    def __init__(self):
        super(IdentityDimension, self).__init__()

    def modified_dimension(self, size, windowsize):
        if size / windowsize == 1:
            yield IdentityDimension()
        else:
            raise ValueError()

    def integer_based_slice(self, index):
        return index


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
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dimensions = getattr(obj, 'dimensions', None)

    # def __array_prepare__(self, obj, context=None):
    #     print 'prepare', self, obj, context
    #     return np.ndarray.__array_prepare__(self, obj, context)
    #
    # def __array_wrap__(self, obj, context=None):
    #     print 'wrap', self, obj, context
    #     return np.ndarray.__array_wrap__(self, obj, context)

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
            return a,
        if isinstance(a, np.ndarray) and a.dtype == np.bool:
            return a,
        try:
            return tuple(a)
        except TypeError:
            return a,

    def __getslice__(self, i, j):
        print 'GET SLICE', i, j
        return self.__getitem__(slice(i, j))

    def __setitem__(self, index, value):
        index = self._tuplify(index)
        # print 'SET ITEM WITH INDEX', index
        indices = tuple(self._compute_indices(index))
        # print 'SET ITEM WITH INDICES', indices
        # print 'PRESENT SHAPE AND DIMS', self.shape, self.dimensions
        super(ArrayWithUnits, self).__setitem__(indices, value)
        # print 'RESULTING SHAPE AND DIMS', self.shape, self.dimensions

    def __getitem__(self, index):

        if isinstance(index, np.ndarray) \
                and index.dtype == np.bool:
            return np.asarray(self)[index]

        if self.ndim == 1 and isinstance(index, int):
            return np.asarray(self)[index]

        # print '============================='
        # print 'dims', self.dimensions
        # print 'shape', self.shape
        # print 'index', index
        index = self._tuplify(index)
        # print 'tuplified', index
        indices = tuple(self._compute_indices(index))
        # print 'indices', indices
        arr = super(ArrayWithUnits, self).__getitem__(indices)
        new_dims = tuple(self._new_dims(index, arr))
        return self.__class__(arr, new_dims)
