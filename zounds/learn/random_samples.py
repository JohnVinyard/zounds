from featureflow import Node, NotEnoughData
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

    def percent_full(self):
        return float(len(self.indices)) / self.nsamples

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

    def get_batch(self, batch_size):
        if batch_size > self.nsamples:
            raise ValueError(
                'Requested {batch_size} samples, but this instance can provide '
                'at maximum {nsamples}'
                    .format(batch_size=batch_size, nsamples=self.nsamples))

        if batch_size > len(self.indices):
            raise ValueError(
                'Requested {batch_size} samples, but this instance only '
                'currently has {n} samples, with a maximum of {nsamples}'
                    .format(
                    batch_size=batch_size,
                    n=len(self.indices),
                    nsamples=self.nsamples))

        # TODO: this would be much more efficient for repeated calls if I
        # instead maintained a sorted set
        indices = np.random.choice(list(self.indices), batch_size)
        return self.arr[indices, ...]


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


class InfiniteSampler(Node):
    def __init__(self, nsamples=None, dtype=None, needs=None):
        super(InfiniteSampler, self).__init__(needs=needs)
        self.reservoir = Reservoir(nsamples, dtype)

    def _process(self, data):
        cls, feature = data

        # compute the total number of samples in our dataset
        _ids = list(cls.database.iter_ids())
        total_samples = sum(
            len(feature(_id=_id, persistence=cls)) for _id in _ids)
        print 'Total samples', total_samples

        while True:
            for _id in _ids:
                # fetch the features from a single document
                x = feature(_id=_id, persistence=cls)

                # compute the contribution this sample makes to the dataset at
                # large
                feature_size = len(x)
                ratio = float(feature_size) / total_samples

                # determine the appropriate number of samples to contribute to
                # the reservoir
                nsamples = int(self.reservoir.nsamples * ratio)

                print 'Contributing', feature_size, ratio, nsamples

                # select an appropriately-sized and random subset of the feature.
                # this will be shuffled again as it is added to the reservoir,
                # but this ensures that samples are drawn evenly from the
                # duration of the sound
                indices = np.random.randint(0, feature_size, nsamples)

                self.reservoir.add(x[indices, ...])

            yield self.reservoir.get()


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
