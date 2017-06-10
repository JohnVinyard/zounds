import base64
import re
import numpy as np

TIMEDELTA_DTYPE_RE = re.compile(r'\[(?P<dtype>[^\]]+)\]')


def encode_timedelta(td):
    dtype = TIMEDELTA_DTYPE_RE.search(str(td.dtype)).groupdict()['dtype']
    return base64.b64encode(td.astype(np.uint64).tostring()), dtype


def decode_timedelta(t):
    if isinstance(t, np.timedelta64):
        return t

    v = np.fromstring(base64.b64decode(t[0]), dtype=np.uint64)[0]
    s = t[1]
    return np.timedelta64(long(v), s)
