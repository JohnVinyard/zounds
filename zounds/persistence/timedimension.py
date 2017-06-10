from zounds.timeseries import TimeDimension
from basedimension import BaseDimensionEncoder, BaseDimensionDecoder
from util import encode_timedelta, decode_timedelta


class TimeDimensionEncoder(BaseDimensionEncoder):

    def __init__(self):
        super(TimeDimensionEncoder, self).__init__(TimeDimension)

    def dict(self, o):
        return dict(
                frequency=encode_timedelta(o.frequency),
                duration=encode_timedelta(o.duration),
                size=o.size)


class TimeDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(TimeDimensionDecoder, self).__init__(TimeDimension)

    def kwargs(self, d):
        return dict(
            frequency=decode_timedelta(d['frequency']),
            duration=decode_timedelta(d['duration']),
            size=d['size'])
