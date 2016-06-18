import re
from zounds.timeseries import Seconds, Picoseconds, TimeSlice


class RangeUnitUnsupportedException(Exception):
    """
    Raised when an HTTP range request is made with a unit not supported by this
    application
    """
    pass


class UnsatisfiableRangeRequestException(Exception):
    """
    Exception raised when an HTTP range request cannot be satisfied for a
    particular resource
    """
    pass


class RangeRequest(object):
    def __init__(self, range_header):
        self.range_header = range_header
        self.re = re.compile(
                r'^(?P<unit>[^=]+)=(?P<start>[^-]+)-(?P<stop>.*?)$')

    def time_slice(self, start, stop):
        start = float(start)
        try:
            stop = float(stop)
        except ValueError:
            stop = None
        duration = \
            None if stop is None else Picoseconds(int(1e12 * (stop - start)))
        start = Picoseconds(int(1e12 * start))
        return TimeSlice(duration, start=start)

    def byte_slice(self, start, stop):
        start = int(start)
        try:
            stop = int(stop)
        except ValueError:
            stop = None
        return slice(start, stop)

    def range(self):
        raw = self.range_header
        if not raw:
            return slice(None)

        m = self.re.match(raw)
        if not m:
            return slice(None)

        units = m.groupdict()['unit']
        start = m.groupdict()['start']
        stop = m.groupdict()['stop']

        if units == 'bytes':
            return self.byte_slice(start, stop)
        elif units == 'seconds':
            return self.time_slice(start, stop)
        else:
            raise RangeUnitUnsupportedException(units)


class ContentRange(object):
    def __init__(self, unit, start, total, stop=None):
        self.unit = unit
        self.total = total
        self.stop = stop
        self.start = start

    @staticmethod
    def from_timeslice(timeslice, total):
        one_second = Seconds(1)
        stop = None
        start = timeslice.start / one_second
        if timeslice.duration is not None:
            stop = start + (timeslice.duration / one_second)
        return ContentRange(
                'seconds',
                timeslice.start / one_second,
                total / one_second,
                stop)

    @staticmethod
    def from_slice(slce, total):
        return ContentRange('bytes', slce.start, total, slce.stop)

    def __str__(self):
        unit = self.unit
        start = self.start
        stop = self.stop or self.total
        total = self.total
        return '{unit} {start}-{stop}/{total}'.format(**locals())
