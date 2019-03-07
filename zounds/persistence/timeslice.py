from .util import encode_timedelta, decode_timedelta


class TimeSliceEncoder(object):
    def __init__(self):
        super(TimeSliceEncoder, self).__init__()

    def dict(self, ts):
        return dict(
            start=encode_timedelta(ts.start),
            duration=encode_timedelta(ts.duration))


class TimeSliceDecoder(object):
    def __init__(self):
        super(TimeSliceDecoder, self).__init__()

    def kwargs(self, d):
        return dict(
            start=decode_timedelta(d['start']),
            duration=decode_timedelta(d['duration']))

