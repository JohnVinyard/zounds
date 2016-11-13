from frequencydimension import \
    FrequencyDimensionEncoder, FrequencyDimensionDecoder
from identitydimension import IdentityDimensionEncoder, IdentityDimensionDecoder
from timedimension import TimeDimensionEncoder, TimeDimensionDecoder


class DimensionEncoder(object):
    encoders = [
        IdentityDimensionEncoder(),
        TimeDimensionEncoder(),
        FrequencyDimensionEncoder()
    ]

    def __init__(self):
        super(DimensionEncoder, self).__init__()

    def encode(self, o):
        for e in self.encoders:
            if e.matches(o):
                return e.encode(o)
        raise NotImplementedError('No matching strategy')


class DimensionDecoder(object):
    decoders = [
        IdentityDimensionDecoder(),
        TimeDimensionDecoder(),
        FrequencyDimensionDecoder()
    ]

    def __init__(self):
        super(DimensionDecoder, self).__init__()

    def decode(self, d):
        for d in self.decoders:
            if d.matches(d):
                return d.decode(d)
        raise NotImplementedError('No matching strategy')
