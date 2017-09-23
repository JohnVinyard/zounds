from featureflow import Node, Feature, DatabaseIterator, BaseModel, \
    NotEnoughData
from featureflow.nmpy import NumpyFeature
from zounds.core import ArrayWithUnits, IdentityDimension
import numpy as np


class Reservoir(object):
    def __init__(self, nsamples, dtype=None):
        super(Reservoir, self).__init__()

        if not isinstance(nsamples, int):
            raise ValueError('nsamples must be an integer')

        if nsamples <= 0:
            raise ValueError('nsamples must be greater than zero')

        self.nsamples = nsamples
        self.arr = None
        self.indices = set()
        self.dtype = dtype

    def _init_arr(self, samples):
        if self.arr is not None:
            return

        shape = (self.nsamples,) + samples.shape[1:]
        self.arr = np.zeros(shape, dtype=self.dtype or samples.dtype)
        try:
            self.arr = ArrayWithUnits(
                self.arr, (IdentityDimension(),) + samples.dimensions[1:])
        except AttributeError:
            pass

    def add(self, samples, indices=None):
        self._init_arr(samples)

        if indices is None:
            indices = np.random.randint(0, self.nsamples, len(samples))

        if len(indices) != len(samples):
            raise ValueError(
                'number of input samples and indices must match'
                ' but they were {samples} and {indices} respectively'
                .format(samples=len(samples), indices=len(indices)))

        self.arr[indices, ...] = samples
        self.indices.update(indices)

    def get(self):
        if len(self.indices) == self.nsamples:
            return self.arr

        return self.arr[sorted(self.indices), ...]


class MultiplexedReservoir(object):
    def __init__(self, nsamples, dtype=None):
        super(MultiplexedReservoir, self).__init__()
        self.dtype = dtype
        self.reservoir = None
        self.nsamples = nsamples

    def _init_dict(self, samples):
        if self.reservoir is not None:
            return

        if self.reservoir is None:
            self.reservoir = dict(
                (k, Reservoir(self.nsamples, dtype=self.dtype))
                for k in samples.iterkeys())

    def _check_sample_keys(self, samples):
        if set(self.reservoir.keys()) != set(samples.keys()):
            raise ValueError(
                'samples should have keys {keys}'
                    .format(keys=self.reservoir.keys()))

    def add(self, samples):
        self._init_dict(samples)
        self._check_sample_keys(samples)

        indices = None
        for k, v in samples.iteritems():
            if indices is None:
                indices = np.random.randint(0, self.nsamples, len(v))
            self.reservoir[k].add(v, indices=indices)

    def get(self):
        return dict((k, v.get()) for k, v in self.reservoir.iteritems())


class ShuffledSamples(Node):
    def __init__(
            self,
            nsamples=None,
            multiplexed=False,
            dtype=None,
            needs=None):

        super(ShuffledSamples, self).__init__(needs=needs)
        self.reservoir = MultiplexedReservoir(nsamples, dtype=dtype) \
            if multiplexed else Reservoir(nsamples, dtype=dtype)

    def _enqueue(self, data, pusher):
        self.reservoir.add(data)

    def _dequeue(self):
        if not self._finalized:
            raise NotEnoughData()
        return self.reservoir.get()


class ReservoirSampler(Node):
    """
    Use reservoir sampling (http://en.wikipedia.org/wiki/Reservoir_sampling) to
    draw a fixed-size set of random samples from a stream of unknown size.

    This is useful when the samples can fit into memory, but the stream cannot.
    """

    def __init__(self, nsamples=None, wrapper=None, needs=None):
        super(ReservoirSampler, self).__init__(needs=needs)
        if wrapper:
            raise DeprecationWarning('wrapper is no longer used or needed')
        self._nsamples = int(nsamples)
        self._r = None
        self._index = 0

    # TODO: What happens if we have filled up all the sample slots and we run
    # out of data?
    def _enqueue(self, data, pusher):
        if self._r is None:
            shape = (self._nsamples,) + data.shape[1:]
            self._r = np.zeros(shape, dtype=data.dtype)
            try:
                self._r = ArrayWithUnits(
                    self._r, (IdentityDimension(),) + data.dimensions[1:])
            except AttributeError:
                # samples were likely a plain numpy array, and not an
                # ArrayWithUnits instance
                pass

        diff = 0
        if self._index < self._nsamples:
            diff = self._nsamples - self._index
            available = len(data[:diff])
            self._r[self._index: self._index + available] = data[:diff]
            self._index += available

        remaining = len(data[diff:])
        if not remaining:
            return
        indices = np.random.random_integers(0, self._index, size=remaining)
        indices = indices[indices < self._nsamples]
        self._r[indices, ...] = data[diff:][range(len(indices))]
        self._index += remaining

    def _dequeue(self):
        if not self._finalized:
            raise NotEnoughData()

        if self._index <= self._nsamples:
            arr = np.asarray(self._r[:self._index])
            np.random.shuffle(arr)
            if isinstance(self._r, ArrayWithUnits):
                arr = ArrayWithUnits(arr, self._r.dimensions)
            return arr

        return self._r


class RequireSignalToNoiseRatio(Node):
    def __init__(self, max_snr=None, needs=None):
        super(RequireSignalToNoiseRatio, self).__init__(needs=needs)
        self._max_snr = max_snr

    def _process(self, data):
        axes = tuple(range(len(data.shape))[1:])
        a = np.abs(data)
        snr = a.mean(axis=axes) / a.std(axis=axes)
        filtered = data[snr < self._max_snr]
        print len(data), len(filtered)
        yield filtered


def random_samples(feature_func, nsamples, store_shuffled=True):
    """
    Return a base class that samples randomly from examples of the data returned
    by feature_func
    """

    class RandomSamples(BaseModel):
        docs = Feature(
            DatabaseIterator,
            func=feature_func,
            store=False)

        patches = NumpyFeature( \
            ReservoirSampler,
            nsamples=nsamples,
            needs=docs,
            store=store_shuffled)

    return RandomSamples
