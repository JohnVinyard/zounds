from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

import pyximport
pyximport.install()
from countbits import *


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def flatten2d(arr):
    ls = len(arr.shape)
    if 1 == ls:
        raise ValueError('Cannot turn 1d array into 2d array')
    elif 2 == ls:
        return arr
    else:
        return arr.reshape((arr.shape[0], np.product(arr.shape[1:])))


def downsampled_shape(shape, factor):
    """
    Return the new shape of an array with shape, once downsampled
    by factor.
    """
    return tuple((np.array(shape) / factor).astype(np.int32))


def downsample(arr, factor, method=np.max):
    """
    Downsample an n-dimensional array

    Parameters
        arr    - the array to be downsampled
        factor - the downsampling factor. If factor is an integer, it is assumed
                 that the array will be downsampled by a constant factor in every
                 dimension. If factor is a tuple, the array will be downsampled
                 by different factors in each dimension. If factor is a tuple
                 whose size is less than the number of dimensions in arr, it
                 is assumed that the downsampling will only be applied to the
                 last len(factor) dimensions of arr.
        method - the method used to combine chunks into single values, e.g.
                 mean, or max.
    """
    if isinstance(factor, int):
        # factor is an integer, so we'll downsample by a constant factor over
        # all dimensions of the input array
        factor = (factor,) * arr.ndim
        lf = arr.ndim
    else:
        # factor is a tuple, so we'll downsample over the last n dimensions of
        # arr by the factors specified
        factor = norm_shape(factor)
        lf = len(factor)
        if lf > arr.ndim:
            raise ValueError( \
                    'The number of factors must be less than or equal to the number of dimensions in arr')

    # the axes over which to apply the reduction
    axes = -np.arange(1, lf + 1)
    # the window size in each dimension of arr
    window_size = ((1,) * (arr.ndim - lf)) + factor
    # get non-overlapping windows whose dimensions are specified by factor
    windowed = sliding_window(arr, window_size, flatten=False)
    # apply the reduction and remove any extraneous dimensions
    return np.apply_over_axes(method, windowed, axes).squeeze()


def safe_log(a):
    """
    Return the element-wise log of an array, checking for negative
    array elements and avoiding divide-by-zero errors.
    """
    if np.any([a < 0]):
        raise ValueError('array contains negative components')

    return np.log(a + 1e-12)


def euclidean_norm(a):
    """
    Return the euclidean norm of each data example
    """
    if 1 == len(a.shape):
        return np.linalg.norm(a)

    return np.sum(np.abs(a) ** 2, axis=-1) ** (1. / 2)


def safe_unit_norm(a):
    """
    Ensure that the vector or vectors have unit norm
    """
    if 1 == len(a.shape):
        n = np.linalg.norm(a)
        if n:
            return a / n
        return a

    norm = np.sum(np.abs(a) ** 2, axis=-1) ** (1. / 2)
    # Dividing by a norm of zero will cause a warning to be issued. Set those
    # values to another number. It doesn't matter what, since we'll be dividing
    # a vector of zeros by the number, and 0 / N always equals 0.
    norm[norm == 0] = -1e12
    return a / norm[:, np.newaxis]


def binarize(arr, thresh=0.5):
    arr[arr < thresh] = 0
    arr[arr >= thresh] = 1
    return arr


def pad(a, desiredlength):
    """
    Pad an n-dimensional numpy array with zeros along the zero-th dimension
    so that it is the desired length.  Return it unchanged if it is greater
    than or equal to the desired length
    """

    if len(a) >= desiredlength:
        return a

    islist = isinstance(a, list)
    a = np.array(a)
    diff = desiredlength - len(a)
    shape = list(a.shape)
    shape[0] = diff

    padded = np.concatenate([a, np.zeros(shape, dtype=a.dtype)])
    return padded.tolist() if islist else padded


def _wpad(l, windowsize, stepsize):
    """
    Parameters
        l - The length of the input array
        windowsize - the size of each window of samples
        stepsize - the number of samples to move the window each step

    Returns
        The length the input array should be so that no samples are leftover
    """
    if l <= windowsize:
        return windowsize

    nsteps = ((l // stepsize) * stepsize)
    overlap = (windowsize - stepsize)
    if overlap:
        return nsteps + overlap

    diff = (l - nsteps)
    left = max(0, windowsize - diff)
    return l + left if diff else l


def _wcut(l, windowsize, stepsize):
    """
    Parameters
        l - The length of the input array
        windowsize - the size of each window of samples
        stepsize - the number of samples to move the window each step

    Returns
        The length the input array should be so that leftover samples are ignored
    """
    end = l - windowsize
    if l <= windowsize:
        return 0, 0
    elif end % stepsize:
        l = windowsize + ((end // stepsize) * stepsize)

    return l, l - (windowsize - stepsize)


def windowed(a, windowsize, stepsize=None, dopad=False):
    """
    Parameters
        a          - the input array to restructure into overlapping windows
        windowsize - the size of each window of samples
        stepsize   - the number of samples to shift the window each step. If not
                     specified, this defaults to windowsize
        dopad      - If false (default), leftover samples are returned seperately.
                     If true, the input array is padded with zeros so that all
                     samples are used.
    """
    if windowsize < 1:
        raise ValueError('windowsize must be greater than or equal to one')

    if stepsize is None:
        stepsize = windowsize

    if stepsize < 1:
        raise ValueError('stepsize must be greater than or equal to one')

    if not a.flags['C_CONTIGUOUS']:
        a = a.copy()

    if windowsize == 1 and stepsize == 1:
        # A windowsize and stepsize of one mean that no windowing is necessary.
        # Return the array unchanged.
        return np.zeros((0,) + a.shape[1:], dtype=a.dtype), a

    if windowsize == 1 and stepsize > 1:
        return np.zeros(0, dtype=a.dtype), a[::stepsize]

    # the original length of the input array
    l = a.shape[0]

    if dopad:
        p = _wpad(l, windowsize, stepsize)
        # pad the array with enough zeros so that there are no leftover samples
        a = pad(a, p)
        # no leftovers; an empty array
        leftover = np.zeros((0,) + a.shape[1:], dtype=a.dtype)
    else:
        # cut the array so that any leftover samples are returned seperately
        c, lc = _wcut(l, windowsize, stepsize)
        leftover = a[lc:]
        a = a[:c]

    if 0 == a.shape[0]:
        return leftover, np.zeros(a.shape, dtype=a.dtype)

    n = 1 + (a.shape[0] - windowsize) // (stepsize)
    s = a.strides[0]
    newshape = (n, windowsize) + a.shape[1:]
    newstrides = (stepsize * s, s) + a.strides[1:]

    out = np.ndarray.__new__( \
            np.ndarray,
            strides=newstrides,
            shape=newshape,
            buffer=a,
            dtype=a.dtype)

    return leftover, out


def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    """

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
                'a.shape, ws and ss must all have the same length. They were %s' % str(
                    ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
                'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides

    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)


class TypeCodes(object):
    _bits = [8, 16, 32, 64]
    _np_types = [np.uint8, np.uint16, np.uint32, np.uint64]
    _type_codes = ['B', 'H', 'L', 'Q']

    @classmethod
    def _fromto(cls, f, t, v):
        return t[f.index(v)]

    @classmethod
    def _whichlist(cls, v):

        if isinstance(v, int):
            return cls._bits

        if isinstance(v, str):
            return cls._type_codes

        if isinstance(v, type) or isinstance(v, np.dtype):
            return cls._np_types

        raise ValueError('%s is not a valid key' % v)

    @classmethod
    def bits(cls, v):
        """
        Get the number of bits from a numpy dtype or typecode
        """
        return cls._fromto(cls._whichlist(v), cls._bits, v)

    @classmethod
    def bytes(cls, v):
        return cls.bits(v) // 8

    @classmethod
    def np_dtype(cls, v):
        """
        Get a numpy dtype from a typecode or number of bits
        """
        return cls._fromto(cls._whichlist(v), cls._np_types, v)

    @classmethod
    def type_code(cls, v):
        """
        Get a typecode from a numpy dtype or a number of bits
        """
        return cls._fromto(cls._whichlist(v), cls._type_codes, v)


class Growable(object):
    """
    A thin wrapper around a numpy array that allows it to be treated like a
    dynamically-sized object.  For numeric types, this should be more memory
    efficient than using a list, since members are represented by contiguous
    blocks of memory instead of PyObjects.
    """

    def __init__(self, data, position=0, growth_rate=1):
        """__init__

        :param data: A numpy array to treat as dynamically resizable. Can have \
        length 0

        :param position: The logical size of the array.  This can be smaller, \
        equal to, or larger than the array passed in as data

        :param growth_rate: The amount by which the array will grow when \
        necessary.  A growth rate of 1 means that the array will double in \
        size each time.
        """
        if growth_rate <= 0:
            raise ValueError('growth_rate must be greater than zero')

        object.__init__(self)
        self._data = data
        self._position = position
        self._growth_rate = growth_rate

    @property
    def data(self):
        """
        Return the numpy array wrapped by this instance.  Note that the array
        may be larger or smaller than logical_size
        """
        return self._data

    @property
    def logical_size(self):
        """
        Return the size of the initialized part of the array.  Values outside
        this boundary are invalid.
        """
        return self._position

    @property
    def physical_size(self):
        """
        The actual size of the wrapped array
        """
        return self._data.shape[0]

    @property
    def logical_data(self):
        """
        Return only the initialized part of the array
        """
        return self._data[:self.logical_size]

    def _tmp_size(self):
        n_items = int(self._data.shape[0] * self._growth_rate)
        # require that n_items is at least one
        return 1 if not n_items else n_items

    def _tmp(self, amt=None):
        n_items = self._tmp_size() if None is amt else amt
        return np.ndarray( \
                (n_items,) + self._data.shape[1:], dtype=self._data.dtype)

    def _grow(self, amt=None):
        new_mem = self._tmp(amt=amt)
        self._data = np.concatenate([self._data, new_mem])

    def append(self, item):
        """
        append a single item to the array, growing the wrapped numpy array
        if necessary
        """
        try:
            self._data[self._position] = item
        except IndexError:
            self._grow()
            self._data[self._position] = item
        self._position += 1
        return self

    def extend(self, items):
        """
        extend the numpy array with multiple items, growing the wrapped array
        if necessary
        """
        items = np.array(items)
        pos = items.shape[0] + self.logical_size
        if pos > self.physical_size:
            # more physical memory is needed than has been allocated so far
            amt = self._tmp_size()
            if self.physical_size + amt < pos:
                # growing the memory by the prescribed amount still won't 
                # make enough room. Allocate exactly as much memory as is needed
                # to accommodate items
                amt = pos - self.physical_size

            # actually grow the array
            self._grow(amt=amt)

        stop = self._position + items.shape[0]
        self._data[self._position: stop] = items
        self._position += items.shape[0]
        return self


def hamming_distance(a, b):
    """
    a - scalar
    b - array of scalars
    """
    return count_bits(a ^ b)


def packed_hamming_distance(a, b):
    """
    Interpret a as a "packed" scalar, i.e. an n-bit number where n may not be
    a power of 2. E.g., a 250-bit number would be represented by 4 64-bit integers.

    Interpret b as an array of "packed" scalars. Its second dimension should be
    the same length as a.
    """
    xored = a ^ b
    bitcount = count_bits(xored.ravel())
    return bitcount.reshape(xored.shape).sum(1)


def jaccard_distance(a, b):
    """
    a - scalar
    b - array of scalars
    """
    intersection = count_bits(a & b)
    union = count_bits(a | b)
    return 1 - (intersection / union)


# TODO: Write tests
class BKTree(object):
    """
    Take advantage of the triangle inequality.  Build a depth-one tree
    where each branch contains the indices of all hashes which are distance n
    from a seed value.

    When querying, take the distance between the query and seed values.  Begin
    with the branch for that distance, and proceed outward.
    """

    def __init__(self, hashes=None, distance=hamming_distance):
        object.__init__(self)
        self.hashes = hashes
        self.distance = distance
        self.seed = self.hashes[np.random.randint(0, len(hashes))]
        self._build()

    def _build(self):
        d = self.distance(self.seed, self.hashes)
        self.tree = dict((dist, np.where(d == dist)[0]) for dist in set(d))

    def append(self, hashes):
        d = self.distance(self.seed, self.hashes)
        tree = dict((dist, np.where(d == dist)[0]) for dist in set(d))
        for k, v in tree.iteritems():
            try:
                self.tree[k] = np.concatenate([v, self.tree[k]])
            except KeyError:
                self.tree[k] = v

    def query(self, query, nresults):
        d = self.distance(query, [self.seed])[0]
        results = np.zeros(nresults, dtype=np.uint32)
        resultspos = 0
        keys = np.array(self.tree.keys())
        ordered = list(keys[np.argsort(np.abs(keys - d))])
        while ordered and resultspos < nresults:
            dist = ordered.pop(0)
            r = self.tree[dist]
            lr = len(r)
            stop = min(nresults, resultspos + lr)
            results[resultspos: stop] = r[:stop - resultspos]
            resultspos += (stop - resultspos)

        return results
