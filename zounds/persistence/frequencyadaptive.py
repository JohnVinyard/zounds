from featureflow import Feature
from .arraywithunits import ArrayWithUnitsDecoder, ArrayWithUnitsEncoder
from zounds.spectral import FrequencyAdaptive


class FrequencyAdaptiveDecoder(ArrayWithUnitsDecoder):
    def __init__(self):
        super(ArrayWithUnitsDecoder, self).__init__()

    def __call__(self, flo):
        raw = super(FrequencyAdaptiveDecoder, self).__call__(flo)
        return FrequencyAdaptive.from_array_with_units(raw)


class FrequencyAdaptiveFeature(Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=ArrayWithUnitsEncoder,
            decoder=FrequencyAdaptiveDecoder(),
            **extractor_args):
        super(FrequencyAdaptiveFeature, self).__init__(
            extractor,
            needs=needs,
            store=store,
            encoder=encoder,
            decoder=decoder,
            key=key,
            **extractor_args)
