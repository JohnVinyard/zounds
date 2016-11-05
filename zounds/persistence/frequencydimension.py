from basedimension import BaseDimensionEncoder, BaseDimensionDecoder
from zounds.spectral import FrequencyBand, FrequencyScale, FrequencyDimension


class FrequencyDimensionEncoder(BaseDimensionEncoder):
    def __init__(self):
        super(FrequencyDimensionEncoder, self).__init__(FrequencyDimension)

    def dict(self, o):
        return dict(
            n_bands=o.scale.n_bands,
            start_hz=o.scale.frequency_band.start_hz,
            stop_hz=o.scale.frequency_band.stop_hz)


class FrequencyDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(FrequencyDimensionDecoder, self).__init__(FrequencyDimension)

    def args(self, d):
        band = FrequencyBand(d['start_hz'], d['stop_hz'])
        scale = FrequencyScale(band, d['n_bands'])
        return scale,

