import numpy as np
from featureflow import Node, NotEnoughData

from zounds.core import ArrayWithUnits
from zounds.nputil import sliding_window
from zounds.timeseries import TimeSlice


def oggvorbis(s):
    """
    This is taken from the ogg vorbis spec
    (http://xiph.org/vorbis/doc/Vorbis_I_spec.html)

    :param s: the total length of the window, in samples
    """
    try:
        s = np.arange(s)
    except TypeError:
        s = np.arange(s[0])

    i = np.sin((s + .5) / len(s) * np.pi) ** 2
    f = np.sin(.5 * np.pi * i)
    return f * (1. / f.max())


class WindowingFunc(object):

    def __init__(self, windowing_func=lambda size: np.ones(size)):
        super(WindowingFunc, self).__init__()
        self.windowing_func = windowing_func

    def _wdata(self, size):
        return self.windowing_func(size)

    def __numpy_ufunc__(self, *args, **kwargs):
        # KLUDGE: This seems really odd, but the mere presence of this
        # numpy-specific magic method seems to serve as a hint to to call
        # this instances __rmul__ implementation, instead of doing element-wise
        # multiplication, as per:
        # http://docs.scipy.org/doc/numpy/reference/arrays.classes.html#numpy.class.__numpy_ufunc__
        raise NotImplementedError()

    def __mul__(self, other):
        size = other.shape[-1]
        return self._wdata(size) * other

    def __rmul__(self, other):
        return self.__mul__(other)


class IdentityWindowingFunc(WindowingFunc):
    def __init__(self):
        super(IdentityWindowingFunc, self).__init__()


class OggVorbisWindowingFunc(WindowingFunc):
    def __init__(self):
        super(OggVorbisWindowingFunc, self).__init__(windowing_func=oggvorbis)


class HanningWindowingFunc(WindowingFunc):
    def __init__(self):
        super(HanningWindowingFunc, self).__init__(windowing_func=np.hanning)


class SlidingWindow(Node):
    def __init__(self, wscheme, wfunc=None, padwith=0, needs=None):
        super(SlidingWindow, self).__init__(needs=needs)
        self._scheme = wscheme
        self._func = wfunc
        self._padwith = padwith
        self._cache = None

    def _first_chunk(self, data):
        if self._padwith:
            padding = np.zeros(
                    (self._padwith,) + data.shape[1:], dtype=data.dtype)
            padding_ts = ArrayWithUnits(padding, data.dimensions)
            return padding_ts.concatenate(data)
        else:
            return data

    def _enqueue(self, data, pusher):
        if self._cache is None:
            self._cache = data
        else:
            self._cache = self._cache.concatenate(data)

    def _dequeue(self):

        duration = TimeSlice(self._scheme.duration)
        frequency = TimeSlice(self._scheme.frequency)

        leftover, arr = self._cache.sliding_window_with_leftovers(
                duration,
                frequency,
                dopad=self._finalized)

        if not arr.size:
            raise NotEnoughData()

        self._cache = leftover

        # BUG: Order matters here (try arr * self._func instead)
        # why does that statement result in __rmul__ being called for each
        # scalar value in arr?
        out = (self._func * arr) if self._func else arr

        return out