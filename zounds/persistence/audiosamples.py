from featureflow import Feature
from zounds.persistence.arraywithunits import \
    ArrayWithUnitsDecoder, ArrayWithUnitsEncoder
from zounds.timeseries import audio_sample_rate, AudioSamples


class AudioSamplesDecoder(ArrayWithUnitsDecoder):
    def __init__(self):
        super(ArrayWithUnitsDecoder, self).__init__()

    def __call__(self, flo):
        raw = super(AudioSamplesDecoder, self).__call__(flo)
        samplerate = audio_sample_rate(raw.dimensions[0].samples_per_second)
        return AudioSamples(raw, samplerate)


class AudioSamplesFeature(Feature):

    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=ArrayWithUnitsEncoder,
            decoder=AudioSamplesDecoder(),
            **extractor_args):
        super(AudioSamplesFeature, self).__init__(
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
