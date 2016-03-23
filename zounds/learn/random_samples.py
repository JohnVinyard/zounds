from featureflow import Node, Feature, DatabaseIterator, BaseModel, NotEnoughData
from featureflow.nmpy import NumpyFeature
from zounds.spectral import NDSlidingWindow
import numpy as np


class ReservoirSampler(Node):
    '''
    Use reservoir sampling (http://en.wikipedia.org/wiki/Reservoir_sampling) to
    draw a fixed-size set of random samples from a stream of unknown size.
    
    This is useful when the samples can fit into memory, but the stream cannot.
    '''

    def __init__(self, needs=None, nsamples=None):
        super(ReservoirSampler, self).__init__(needs=needs)
        self._nsamples = nsamples
        self._r = None
        self._index = 0

    # TODO: What happens if we have filled up all the sample slots and we run
    # out of data?
    def _enqueue(self, data, pusher):
        if self._r is None:
            shape = (self._nsamples,) + data.shape[1:]
            self._r = np.zeros(shape, dtype=data.dtype)

        diff = 0
        if self._index < self._nsamples:
            diff = self._nsamples - self._index
            available = len(data[:diff])
            self._r[self._index: self._index + available] = data[:diff]
            self._index += available

        remaining = len(data[diff:])
        if not remaining: return
        indices = np.random.random_integers(0, self._index, size=remaining)
        indices = indices[indices < self._nsamples]
        self._r[indices] = data[diff:][range(len(indices))]
        self._index += remaining

    def _dequeue(self):
        if not self._finalized:
            raise NotEnoughData()

        if self._index <= self._nsamples:
            arr = self._r[:self._index]
            np.random.shuffle(arr)
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
    '''
    Return a base class that samples randomly from examples of the data returned
    by feature_func
    '''

    class RandomSamples(BaseModel):
        docs = Feature( \
                DatabaseIterator,
                func=feature_func,
                store=False)

        patches = NumpyFeature( \
                ReservoirSampler,
                nsamples=nsamples,
                needs=docs,
                store=store_shuffled)

    return RandomSamples


def random_patches( \
        feature_func,
        windowsize,
        nsamples,
        stepsize=None,
        store_shuffled=True):
    '''
    Return a base class that samples random, n-dimensional patches from a 
    feature, e.g., 10x10 patches from spectrograms
    '''

    if stepsize is None:
        stepsize = (1,) * len(windowsize)

    class RandomPatches(BaseModel):
        docs = Feature( \
                DatabaseIterator,
                func=feature_func,
                store=False)

        windowed = NumpyFeature( \
                NDSlidingWindow,
                windowsize=windowsize,
                stepsize=stepsize,
                needs=docs,
                store=False)

        patches = NumpyFeature( \
                ReservoirSampler,
                nsamples=nsamples,
                needs=windowed,
                store=store_shuffled)

    return RandomPatches
