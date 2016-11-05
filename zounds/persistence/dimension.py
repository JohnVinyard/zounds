from identitydimension import IdentityDimensionEncoder, IdentityDimensionDecoder
from timedimension import TimeDimensionEncoder, TimeDimensionDecoder
from frequencydimension import \
    FrequencyDimensionEncoder, FrequencyDimensionDecoder
import json


class DimensionsEncoder(json.JSONEncoder):
    encoders = [
        IdentityDimensionEncoder(),
        TimeDimensionEncoder(),
        FrequencyDimensionEncoder()
    ]


class DimensionsDecoder(json.JSONDecoder):
    decoders = [
        IdentityDimensionDecoder(),
        TimeDimensionDecoder(),
        FrequencyDimensionDecoder()
    ]
    pass
