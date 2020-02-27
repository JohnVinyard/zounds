import base64
import re
import numpy as np
import inspect

TIMEDELTA_DTYPE_RE = re.compile(r'\[(?P<dtype>[^\]]+)\]')


def encode_timedelta(td):
    dtype = TIMEDELTA_DTYPE_RE.search(str(td.dtype)).groupdict()['dtype']
    # base64 encoded value, encoded as a utf-8
    encoded = base64.b64encode(td.astype(np.uint64).tostring()).decode()
    return encoded, dtype


def decode_timedelta(t):
    try:
        v = np.frombuffer(base64.b64decode(t[0]), dtype=np.uint64)[0]
        s = t[1]
        return np.timedelta64(int(v), s)
    except IndexError:
        return t


def extract_init_args(instance):
    """
    Given an instance, and under the assumption that member variables have the
    same name as the __init__ arguments, extract the arguments so they can
    be used to reconstruct the instance when deserializing
    """
    cls = instance.__class__
    args = [x for x in inspect.getargspec(cls.__init__).args if x != 'self']
    return [instance.__dict__[key] for key in args]
