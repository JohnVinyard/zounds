from basedimension import BaseDimensionEncoder, BaseDimensionDecoder
from zounds.spectral import FrequencyBand, FrequencyDimension
import zounds.spectral


class FrequencyDimensionEncoder(BaseDimensionEncoder):
    def __init__(self):
        super(FrequencyDimensionEncoder, self).__init__(FrequencyDimension)

    def dict(self, o):
        return dict(
                n_bands=o.scale.n_bands,
                start_hz=o.scale.frequency_band.start_hz,
                stop_hz=o.scale.frequency_band.stop_hz,
                name=o.scale.__class__.__name__)


class FrequencyDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(FrequencyDimensionDecoder, self).__init__(FrequencyDimension)

    def args(self, d):
        band = FrequencyBand(d['start_hz'], d['stop_hz'])
        # KLUDGE: This assumes that all FrequencyScale-derived classes live in
        # the zounds.spectral module
        scale_class = getattr(zounds.spectral, d['name'])
        scale = scale_class(band, d['n_bands'])
        return scale,
