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
    """
    `WindowingFunc` is mostly a convenient wrapper around `numpy's handy
    windowing functions
    <https://docs.scipy.org/doc/numpy/reference/routines.window.html>`_, or any
    function that takes a size parameter and returns a numpy array-like object.

    A `WindowingFunc` instance can be multiplied with a nother array of any size.

    Args:
        windowing_func (function): A function that takes a size parameter, and
            returns a numpy array-like object

    Examples:
        >>> from zounds import WindowingFunc
        >>> import numpy as np
        >>> wf = WindowingFunc(lambda size: np.hanning(size))
        >>> np.ones(5) *  wf
        array([ 0. ,  0.5,  1. ,  0.5,  0. ])
        >>> np.ones(10) * wf
        array([ 0.        ,  0.11697778,  0.41317591,  0.75      ,  0.96984631,
                0.96984631,  0.75      ,  0.41317591,  0.11697778,  0.        ])

    See Also:
        :class:`~IdentityWindowingFunc`
        :class:`~zounds.spectral.OggVorbisWindowingFunc`
        :class:`~zounds.spectral.HanningWindowingFunc`
    """
    def __init__(self, windowing_func=None):
        super(WindowingFunc, self).__init__()
        self.windowing_func = windowing_func

    def _wdata(self, size):
        if self.windowing_func is None:
            return None
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
        wdata = self._wdata(size)
        if wdata is None:
            return other
        return wdata * other

    def __rmul__(self, other):
        return self.__mul__(other)


class IdentityWindowingFunc(WindowingFunc):
    """
    An identity windowing function
    """
    def __init__(self):
        super(IdentityWindowingFunc, self).__init__()


class OggVorbisWindowingFunc(WindowingFunc):
    """
    The windowing function described in the `ogg vorbis specification
    <https://xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-230001.3.2>`_
    """
    def __init__(self):
        super(OggVorbisWindowingFunc, self).__init__(windowing_func=oggvorbis)


class HanningWindowingFunc(WindowingFunc):
    """
    A hanning window function
    """
    def __init__(self):
        super(HanningWindowingFunc, self).__init__(windowing_func=np.hanning)


class SlidingWindow(Node):
    """
    `SlidingWindow` is a processing node that provides a very common precursor
    to many frequency domain transforms: a lapped and windowed view of the time-
    domain signal.

    Args:
        wscheme (SampleRate): a sample rate that describes the frequency and
            duration af the sliding window
        wfunc (WindowingFunc): a windowing function to apply to each frame
        needs (Node): A processing node on which this node relies for its data.
            This will generally be a time-domain signal

    Here's how you'd typically see :class:`SlidingWindow` used in a processing
    graph

    .. code:: python

        import zounds

        Resampled = zounds.resampled(resample_to=zounds.SR11025())

        @zounds.simple_in_memory_settings
        class Sound(Resampled):
            windowed = zounds.ArrayWithUnitsFeature(
                zounds.SlidingWindow,
                needs=Resampled.resampled,
                wscheme=zounds.SampleRate(
                    frequency=zounds.Milliseconds(250),
                    duration=zounds.Milliseconds(500)),
                wfunc=zounds.OggVorbisWindowingFunc(),
                store=True)


        synth = zounds.SineSynthesizer(zounds.SR44100())
        samples = synth.synthesize(zounds.Seconds(5), [220., 440., 880.])

        # process the audio, and fetch features from our in-memory store
        _id = Sound.process(meta=samples.encode())
        sound = Sound(_id)

        print sound.windowed.dimensions[0]
        # TimeDimension(f=0.250068024879, d=0.500045346811)
        print sound.windowed.dimensions[1]
        # TimeDimension(f=9.0702947e-05, d=9.0702947e-05)

    See Also:
        :class:`~zounds.spectral.WindowingFunc`
        :class:`~zounds.timeseries.SampleRate`
    """
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