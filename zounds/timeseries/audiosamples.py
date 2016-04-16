from featureflow import Node, Feature, Decoder
from timeseries import \
    ConstantRateTimeSeries, ConstantRateTimeSeriesMetadata
from samplerate import AudioSampleRate, audio_sample_rate
import numpy as np
from duration import Seconds
from soundfile import SoundFile
from io import BytesIO


class AudioSamples(ConstantRateTimeSeries):
    def __new__(cls, array, samplerate):
        if not isinstance(samplerate, AudioSampleRate):
            raise TypeError('samplerate should be an AudioSampleRate instance')
        return ConstantRateTimeSeries.__new__(
                cls, array, samplerate.frequency, samplerate.duration)

    @classmethod
    def from_example(cls, arr, example):
        return cls(arr, example.samplerate)

    @property
    def channels(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]

    @property
    def samplerate(self):
        return audio_sample_rate(self.samples_per_second)

    @property
    def mono(self):
        if self.channels == 1:
            return self
        return AudioSamples(self.sum(axis=1) * 0.5, self.samplerate)

    def encode(self, flo=None, fmt='WAV', subtype='PCM_16'):
        flo = flo or BytesIO()
        with SoundFile(
                flo,
                mode='w',
                channels=self.channels,
                format=fmt,
                subtype=subtype,
                samplerate=self.samples_per_second) as f:
            f.write(self)
        flo.seek(0)
        return flo


class AudioSamplesEncoder(Node):
    content_type = 'application/octet-stream'

    def __init__(self, needs=None):
        super(AudioSamplesEncoder, self).__init__(needs=needs)
        self.metadata = None

    def _process(self, data):
        if not self.metadata:
            self.metadata = ConstantRateTimeSeriesMetadata \
                .from_timeseries(data)
            packed = self.metadata.pack()
            yield packed

        encoded = data.tostring()
        yield encoded


def _np_from_buffer(b, shape, dtype, freq, duration):
    f = np.frombuffer if len(b) else np.fromstring
    shape = tuple(int(x) for x in shape)
    f = f(b, dtype=dtype).reshape(shape)
    samples_per_second = Seconds(1) / freq
    samplerate = audio_sample_rate(int(samples_per_second))
    return AudioSamples(f, samplerate)


class GreedyAudioSamplesDecoder(Decoder):
    def __init__(self):
        super(GreedyAudioSamplesDecoder, self).__init__()

    def __call__(self, flo):
        metadata, bytes_read = ConstantRateTimeSeriesMetadata.unpack(flo)

        leftovers = flo.read()
        leftover_bytes = len(leftovers)
        first_dim = leftover_bytes / metadata.totalsize
        dim = (first_dim,) + metadata.shape
        out = _np_from_buffer(
                leftovers,
                dim,
                metadata.dtype,
                metadata.frequency,
                metadata.duration)
        return out

    def __iter__(self, flo):
        yield self(flo)


class AudioSamplesFeature(Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=AudioSamplesEncoder,
            decoder=GreedyAudioSamplesDecoder(),
            **extractor_args):
        super(AudioSamplesFeature, self).__init__(
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
