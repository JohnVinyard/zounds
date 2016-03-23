from featureflow import Node, NotEnoughData
import numpy as np
from zounds.nputil import windowed, sliding_window
from zounds.timeseries import ConstantRateTimeSeries


def oggvorbis(s):
    '''
    This is taken from the ogg vorbis spec 
    (http://xiph.org/vorbis/doc/Vorbis_I_spec.html)

    s is the total length of the window, in samples
    '''
    try:
        s = np.arange(s)
    except TypeError:
        s = np.arange(s[0])

    i = np.sin((s + .5) / len(s) * np.pi) ** 2
    f = np.sin(.5 * np.pi * i)
    return f * (1. / f.max())


class WindowingFunc(object):
    def _wdata(self, size):
        return np.ones(size)

    def __mul__(self, other):
        size = other.shape[1:]
        return self._wdata(size) * other

    def __rmul__(self, other):
        return self.__mul__(other)


class IdentityWindowingFunc(WindowingFunc):
    def __init__(self):
        super(IdentityWindowingFunc, self).__init__()


class OggVorbisWindowingFunc(WindowingFunc):
    def __init__(self):
        super(OggVorbisWindowingFunc, self).__init__()

    def _wdata(self, size):
        return oggvorbis(size)


class SlidingWindow(Node):
    def __init__(self, wscheme, wfunc=None, padwith=0, needs=None):
        super(SlidingWindow, self).__init__(needs=needs)
        self._scheme = wscheme
        self._func = wfunc or IdentityWindowingFunc()
        self._padwith = padwith
        self._cache = None

    def _first_chunk(self, data):
        padding = np.zeros((self._padwith,) + data.shape[1:], dtype=data.dtype)
        padding_ts = ConstantRateTimeSeries( \
                padding,
                data.frequency,
                data.duration)
        return padding_ts.concatenate(data)

    def _enqueue(self, data, pusher):
        if self._cache is None:
            self._cache = data
            # BUG: I Think this may only work in cases where frequency and
            # duration are the same
            self._windowsize = \
                int((self._scheme.duration - data.overlap) / data.frequency)
            self._stepsize = int(self._scheme.frequency / data.frequency)
        else:
            self._cache = np.concatenate([self._cache, data])

    def _dequeue(self):
        leftover, arr = windowed( \
                self._cache,
                self._windowsize,
                self._stepsize,
                dopad=self._finalized)

        self._cache = leftover

        if not arr.size:
            raise NotEnoughData()

        # BUG: Order matters here (try arr * self._func instead)
        # why does that statement result in __rmul__ being called for each
        # scalar value in arr?
        out = (self._func * arr) if self._func else arr
        out = ConstantRateTimeSeries( \
                out, self._scheme.frequency, self._scheme.duration)
        return out


# KLUDGE: This extractor works when trying to get random patches over the whole
# database, but doesn't work if we were trying to get a sliding window over, say,
# the bark bands of a single sound.  For that to work, we'd have to implement
# enqueue and dequeue methods that keep leftovers around between calls.
#
# If we're processing bark bands of a single sound all at once, we expect the
# sliding window to move over the time dimension first, and then over the frequency
# dimension.  Since each incoming chunk is of inderteminate size, we're left with
# nonsense ouput. This problem could be addressed by transposing the incoming data,
# so that the frequency dimension is traversed first, and then the time dimension.
class NDSlidingWindow(Node):
    def __init__(self, windowsize=None, stepsize=None, needs=None):
        super(NDSlidingWindow, self).__init__(needs=needs)
        self._ws = windowsize
        self._ss = stepsize

    def _process(self, data):
        try:
            yield sliding_window(data, self._ws, self._ss)
        except ValueError:
            pass
