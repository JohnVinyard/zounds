from axis import ArrayWithUnits
from featureflow import NumpyEncoder, BaseNumpyDecoder, Feature
from dimensions import IdentityDimension


class ArrayWithUnitsEncoder(NumpyEncoder):
    def __init__(self, needs=None):
        super(ArrayWithUnitsEncoder, self).__init__(needs=needs)

    def _prepare_data(self, data):
        return data


class ArrayWithUnitsDecoder(BaseNumpyDecoder):
    def __init__(self):
        super(ArrayWithUnitsDecoder, self).__init__()

    def _wrap_array(self, raw, metadata):
        dims = (IdentityDimension(), ) * raw.ndim
        return ArrayWithUnits(raw, dims)


class ArrayWithUnitsFeature(Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=ArrayWithUnitsEncoder,
            decoder=ArrayWithUnitsDecoder(),
            **extractor_args):
        super(ArrayWithUnitsFeature, self).__init__(
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
