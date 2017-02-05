import numpy as np
from duration import Picoseconds, Seconds
from samplerate import SampleRate
from zounds.core import Dimension


class TimeSlice(object):
    def __init__(self, duration=None, start=None):
        super(TimeSlice, self).__init__()

        if duration is not None and not isinstance(duration, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=duration.__class__))

        if start is not None and not isinstance(start, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=start.__class__))

        self.duration = duration
        self.start = start or Picoseconds(0)

    def __add__(self, other):
        return TimeSlice(self.duration, start=self.start + other)

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def end(self):
        return self.start + self.duration

    def __lt__(self, other):
        try:
            return self.start.__lt__(other.start)
        except AttributeError:
            return self.start.__lt__(other)

    def __gt__(self, other):
        try:
            return self.start.__gt__(other.start)
        except AttributeError:
            return self.start.__get__(other)

    def __le__(self, other):
        try:
            return self.start.__le__(other.start)
        except AttributeError:
            return self.start.__le__(other)

    def __ge__(self, other):
        try:
            return self.start.__ge__(other.start)
        except AttributeError:
            return self.start.__ge__(other)

    def __and__(self, other):
        delta = max(
                Picoseconds(0),
                min(self.end, other.end) - max(self.start, other.start))
        return TimeSlice(delta)

    def __contains__(self, other):
        print other
        if isinstance(other, np.timedelta64):
            return self.start < other < self.end
        if isinstance(other, TimeSlice):
            return other.start > self.start and other.end < self.end
        raise ValueError

    def __eq__(self, other):
        return self.start == other.start and self.duration == other.duration

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        dur = self.duration / Seconds(1) if self.duration is not None else None
        return '{cls}(start = {start}, duration = {duration})'.format(
                cls=self.__class__.__name__,
                start=self.start / Seconds(1),
                duration=dur)

    def __str__(self):
        return self.__repr__()


class TimeDimension(Dimension):
    def __init__(self, frequency=None, duration=None, size=None):
        super(TimeDimension, self).__init__()
        self.size = size
        if not isinstance(frequency, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=frequency.__class__))

        if duration is not None and not isinstance(duration, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(
                    t=np.timedelta64, t2=duration.__class__))
        self.duration = duration or frequency
        self.frequency = frequency

    def __str__(self):
        fs = self.frequency / Picoseconds(int(1e12))
        ds = self.duration / Picoseconds(int(1e12))
        return 'TimeDimension(f={fs}, d={ds})'.format(**locals())

    def __repr__(self):
        return self.__str__()

    @property
    def samplerate(self):
        return SampleRate(self.frequency, self.duration)

    @property
    def overlap(self):
        return self.samplerate.overlap

    @property
    def duration_in_seconds(self):
        return self.duration / Picoseconds(int(1e12))

    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)

    @property
    def span(self):
        overlap = self.duration - self.frequency
        return TimeSlice((self.size * self.frequency) + overlap)

    @property
    def end(self):
        return self.span.end

    @property
    def end_seconds(self):
        return self.end / Picoseconds(int(1e12))

    def modified_dimension(self, size, windowsize, stepsize=None):
        stepsize = stepsize or windowsize
        yield TimeDimension(
                self.frequency * stepsize,
                (self.frequency * windowsize) + self.overlap)
        yield self

    def metaslice(self, index, size):
        return TimeDimension(self.frequency, self.duration, size)

    def integer_based_slice(self, ts):
        if not isinstance(ts, TimeSlice):
            return ts

        diff = self.duration - self.frequency
        start_index = \
            max(0, np.floor((ts.start - diff) / self.frequency))
        end = self.end if ts.duration is None else ts.end

        # KLUDGE: This is basically arbitrary, but the motivation is that we'd
        # like to differentiate between cases where the slice
        # actually/intentionally overlaps a particular sample, and cases where
        # the slice overlaps the sample by a tiny amount, due to rounding or
        # lack of precision (e.g. Seconds(1) / SR44100().frequency).
        ratio = np.round(end / self.frequency, 2)

        stop_index = np.ceil(ratio)
        return slice(start_index, stop_index)

    def __eq__(self, other):
        return \
            self.frequency == other.frequency \
            and self.duration == other.duration
