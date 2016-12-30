from zounds.timeseries import TimeDimension
import numpy as np
from basedimension import BaseDimensionEncoder, BaseDimensionDecoder
import re
import base64


class TimeDimensionEncoder(BaseDimensionEncoder):
    DTYPE_RE = re.compile(r'\[(?P<dtype>[^\]]+)\]')

    def __init__(self):
        super(TimeDimensionEncoder, self).__init__(TimeDimension)

    def _encode_timedelta(self, td):
        dtype = self.DTYPE_RE.search(str(td.dtype)).groupdict()['dtype']
        return base64.b64encode(td.astype(np.uint64).tostring()), dtype

    def dict(self, o):
        return dict(
                frequency=self._encode_timedelta(o.frequency),
                duration=self._encode_timedelta(o.duration),
                size=o.size)


class TimeDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(TimeDimensionDecoder, self).__init__(TimeDimension)

    def _decode_timedelta(self, t):
        if isinstance(t, np.timedelta64):
            return t

        v = np.fromstring(base64.b64decode(t[0]), dtype=np.uint64)[0]
        s = t[1]
        return np.timedelta64(long(v), s)

    def kwargs(self, d):
        return dict(
            frequency=self._decode_timedelta(d['frequency']),
            duration=self._decode_timedelta(d['duration']),
            size=d['size'])
