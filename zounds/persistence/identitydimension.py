from basedimension import BaseDimensionEncoder, BaseDimensionDecoder
from zounds.core import IdentityDimension


class IdentityDimensionEncoder(BaseDimensionEncoder):
    def __init__(self):
        super(IdentityDimensionEncoder, self).__init__(IdentityDimension)

    def dict(self, o):
        return dict()


class IdentityDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(IdentityDimensionDecoder, self).__init__(IdentityDimension)
