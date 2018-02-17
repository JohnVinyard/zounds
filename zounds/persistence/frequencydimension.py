from basedimension import BaseDimensionEncoder, BaseDimensionDecoder
from zounds.spectral import \
    FrequencyBand, FrequencyDimension, LinearScale, GeometricScale, \
    ExplicitScale, ExplicitFrequencyDimension, BarkScale, MelScale, ChromaScale


class ScaleEncoderDecoder(object):
    def __init__(self):
        super(ScaleEncoderDecoder, self).__init__()

    def can_encode(self, scale):
        raise NotImplementedError()

    def can_decode(self, d):
        raise NotImplementedError()

    def decode(self, d):
        raise NotImplementedError()

    def encode(self, scale):
        raise NotImplementedError()


class GenericScaleEncoderDecoder(ScaleEncoderDecoder):
    def __init__(self, cls):
        super(GenericScaleEncoderDecoder, self).__init__()
        self.cls = cls

    def can_encode(self, scale):
        return isinstance(scale, self.cls)

    def can_decode(self, d):
        return d['name'] == self.cls.__name__

    def encode(self, scale):
        return dict(
            start_hz=scale.frequency_band.start_hz,
            stop_hz=scale.frequency_band.stop_hz,
            n_bands=scale.n_bands,
            name=scale.__class__.__name__)

    def _decode_frequency_band(self, d):
        return FrequencyBand(d['start_hz'], d['stop_hz'])

    def _decode_args(self, d):
        return self._decode_frequency_band(d), d['n_bands']

    def decode(self, d):
        return self.cls(*self._decode_args(d))


class BarkScaleEncoderDecoder(GenericScaleEncoderDecoder):
    def __init__(self):
        super(BarkScaleEncoderDecoder, self).__init__(BarkScale)


class MelScaleEncoderDecoder(GenericScaleEncoderDecoder):
    def __init__(self):
        super(MelScaleEncoderDecoder, self).__init__(MelScale)


class ChromaScaleEncoderDecoder(GenericScaleEncoderDecoder):
    def __init__(self):
        super(ChromaScaleEncoderDecoder, self).__init__(ChromaScale)

    def _decode_args(self, d):
        return self._decode_frequency_band(d),


class LinearScaleEncoderDecoder(ScaleEncoderDecoder):
    def __init__(self):
        super(LinearScaleEncoderDecoder, self).__init__()

    def can_encode(self, scale):
        return isinstance(scale, LinearScale)

    def can_decode(self, d):
        return d['name'] == LinearScale.__name__

    def encode(self, scale):
        return dict(
            start_hz=scale.frequency_band.start_hz,
            stop_hz=scale.frequency_band.stop_hz,
            n_bands=scale.n_bands,
            name=scale.__class__.__name__,
            always_even=scale.always_even)

    def decode(self, d):
        band = FrequencyBand(d['start_hz'], d['stop_hz'])
        return LinearScale(band, d['n_bands'], always_even=d['always_even'])


class GeometricScaleEncoderDecoder(ScaleEncoderDecoder):
    def __init__(self):
        super(GeometricScaleEncoderDecoder, self).__init__()

    def can_encode(self, scale):
        return isinstance(scale, GeometricScale)

    def can_decode(self, d):
        return d['name'] == GeometricScale.__name__

    def encode(self, scale):
        return dict(
            start_center_hz=scale.start_center_hz,
            stop_center_hz=scale.stop_center_hz,
            bandwidth_ratio=scale.bandwidth_ratio,
            n_bands=scale.n_bands,
            name=GeometricScale.__name__,
            always_even=scale.always_even)

    def decode(self, d):
        return GeometricScale(
            d['start_center_hz'],
            d['stop_center_hz'],
            d['bandwidth_ratio'],
            d['n_bands'],
            d['always_even'])


class ExplicitScaleEncoderDecoder(ScaleEncoderDecoder):
    def __init__(self):
        super(ExplicitScaleEncoderDecoder, self).__init__()

    def can_encode(self, scale):
        return isinstance(scale, ExplicitScale)

    def can_decode(self, d):
        return d['name'] == ExplicitScale.__name__

    def encode(self, scale):
        bands = [(b.start_hz, b.stop_hz) for b in scale]
        return dict(bands=bands, name=ExplicitScale.__name__)

    def decode(self, d):
        bands = [FrequencyBand(*b) for b in d['bands']]
        return ExplicitScale(bands)


class CompositeScaleEncoderDecoder(object):
    strategies = [
        LinearScaleEncoderDecoder(),
        GeometricScaleEncoderDecoder(),
        ExplicitScaleEncoderDecoder(),
        BarkScaleEncoderDecoder(),
        MelScaleEncoderDecoder(),
        ChromaScaleEncoderDecoder()
    ]

    def __init__(self):
        super(CompositeScaleEncoderDecoder, self).__init__()

    def _find_encoder(self, scale):
        try:
            return next(s for s in self.strategies if s.can_encode(scale))
        except StopIteration:
            raise NotImplementedError('No suitable encoder found')

    def _find_decoder(self, d):
        try:
            return next(s for s in self.strategies if s.can_decode(d))
        except StopIteration:
            raise NotImplementedError('No suitable decoder found')

    def encode(self, scale):
        return self._find_encoder(scale).encode(scale)

    def decode(self, d):
        return self._find_decoder(d).decode(d)


class FrequencyDimensionEncoder(BaseDimensionEncoder):
    def __init__(self):
        super(FrequencyDimensionEncoder, self).__init__(FrequencyDimension)
        self.scale_encoder = CompositeScaleEncoderDecoder()

    def dict(self, freq_dim):
        return self.scale_encoder.encode(freq_dim.scale)


class FrequencyDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(FrequencyDimensionDecoder, self).__init__(FrequencyDimension)
        self.scale_decoder = CompositeScaleEncoderDecoder()

    def args(self, d):
        return self.scale_decoder.decode(d),


class ExplicitFrequencyDimensionEncoder(BaseDimensionEncoder):
    def __init__(self):
        super(ExplicitFrequencyDimensionEncoder, self).__init__(
            ExplicitFrequencyDimension)
        self.scale_encoder = CompositeScaleEncoderDecoder()

    def dict(self, freq_dim):
        d = self.scale_encoder.encode(freq_dim.scale)
        slices = [(s.start, s.stop) for s in freq_dim.slices]
        d.update(slices=slices)
        return d


class ExplicitFrequencyDimensionDecoder(BaseDimensionDecoder):
    def __init__(self):
        super(ExplicitFrequencyDimensionDecoder, self).__init__(
            ExplicitFrequencyDimension)
        self.scale_decoder = CompositeScaleEncoderDecoder()

    def args(self, d):
        scale = self.scale_decoder.decode(d)
        slices = [slice(start, stop) for (start, stop) in d['slices']]
        return scale, slices
