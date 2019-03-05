from .frequencydimension import \
    FrequencyDimensionEncoder, FrequencyDimensionDecoder, \
    ExplicitFrequencyDimensionEncoder, ExplicitFrequencyDimensionDecoder
from .identitydimension import IdentityDimensionEncoder, IdentityDimensionDecoder
from .timedimension import TimeDimensionEncoder, TimeDimensionDecoder


class DimensionEncoder(object):
    encoders = [
        IdentityDimensionEncoder(),
        TimeDimensionEncoder(),
        FrequencyDimensionEncoder(),
        ExplicitFrequencyDimensionEncoder()
    ]

    def __init__(self):
        super(DimensionEncoder, self).__init__()

    def encode(self, o):
        for dim in o:
            for encoder in self.encoders:
                if encoder.matches(dim):
                    yield encoder.encode(dim)
                    break
            else:
                raise NotImplementedError(
                        'No matching strategy for {dim}'.format(**locals()))


class DimensionDecoder(object):
    decoders = [
        IdentityDimensionDecoder(),
        TimeDimensionDecoder(),
        FrequencyDimensionDecoder(),
        ExplicitFrequencyDimensionDecoder()
    ]

    def __init__(self):
        super(DimensionDecoder, self).__init__()

    def decode(self, d):
        for dim in d:
            for decoder in self.decoders:
                if decoder.matches(dim):
                    yield decoder.decode(dim)
                    break
            else:
                raise NotImplementedError(
                        'No matching strategy for {dim}'.format(**locals()))
