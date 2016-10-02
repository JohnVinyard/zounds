import numpy as np
from zounds.nputil import sliding_window
from zounds.util import tuplify
from itertools import izip_longest

class Dimension(object):
    """
    Class representing one dimension of a numpy array, making custom slices
    (e.g., time spans or frequency bands) possible
    """

    def __init__(self):
        super(Dimension, self).__init__()

    def modified_dimension(self, size, windowsize):
        raise NotImplementedError()

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
        # think about: Ellipsis, full slices,
        for i, sl in enumerate(index):
            try:
                yield self.dimensions[i].integer_based_slice(sl)
            except AttributeError:
                yield sl

    def __setitem__(self, index, value):
        index = tuplify(index)
        indices = tuple(self._compute_indices(index))
        super(ArrayWithUnits, self).__setitem__(indices, value)

    def _new_dims(self, index):
        for i, dim in izip_longest(index, self.dimensions):
            if isinstance(i, int):
                continue
            yield dim

    def __getitem__(self, index):
        index = tuplify(index)
        print index
        indices = tuple(self._compute_indices(index))
        print indices
        arr = super(ArrayWithUnits, self).__getitem__(indices)
        new_dims = tuple(self._new_dims(index))
        print arr.shape, new_dims
        return self.__class__(arr, new_dims)
