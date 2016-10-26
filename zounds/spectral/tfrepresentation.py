import numpy as np
import featureflow as ff
from zounds.timeseries import \
    ConstantRateTimeSeries, ConstantRateTimeSeriesMetadata, TimeDimension
from zounds.core import Dimension, IdentityDimension
import frequencyscale


class FrequencyDimension(Dimension):
    def __init__(self, scale):
        super(FrequencyDimension, self).__init__()
        self.scale = scale

    def modified_dimension(self, size, windowsize):
        raise NotImplementedError()

    def metaslice(self, index, size):
        print 'OK', index, size
        return FrequencyDimension(self.scale[index])

    def integer_based_slice(self, index):
        if not isinstance(index, frequencyscale.FrequencyBand):
            return index

        return self.scale.get_slice(index)


class TimeFrequencyRepresentation(ConstantRateTimeSeries):
    """
    A class that encapsulates time-frequency representation data.  The first
    axis represents time, the second axis represents frequency, and any
    subsequent axes contain multidimensional data about time-frequency positions
    """

    def __new__(cls, arr, frequency=None, duration=None, scale=None):
        if len(arr.shape) < 2:
            raise ValueError('arr must be at least 2D')

        if isinstance(frequency, tuple):
            print frequency, arr.shape
            # KLUDGE: This check is necessary for an initial, incremental
            # refactoring, and should be removed once there are some nice,
            # ArrayWithUnits-derived classes that just work
            scale = frequency[1].scale

        if len(scale) != arr.shape[1]:
            print 'DEBUG', scale, scale.n_bands, arr.shape
            raise ValueError('scale must have same size as dimension 2')

        if isinstance(frequency, tuple):
            # KLUDGE: This check is necessary for an initial, incremental
            # refactoring, and should be removed once there are some nice,
            # ArrayWithUnits-derived classes that just work
            dims = frequency
        else:
            dims = (TimeDimension(frequency, duration, len(arr)),
                    FrequencyDimension(scale))
            dims = dims + \
                   tuple(map(lambda x: IdentityDimension(), arr.shape[2:]))
        obj = ConstantRateTimeSeries.__new__(cls, arr, dims)
        return obj

    def kwargs(self, **kwargs):
        return super(TimeFrequencyRepresentation, self).kwargs(
                scale=self.scale, **kwargs)

    @classmethod
    def from_example(cls, arr, example):
        return cls(
                arr,
                frequency=example.frequency,
                duration=example.duration,
                scale=example.scale)

    @property
    def scale(self):
        return self.dimensions[1].scale

        # def __array_finalize__(self, obj):
        #     super(TimeFrequencyRepresentation, self).__array_finalize__(obj)
        #     if obj is None:
        #         return
        #     self.scale = getattr(obj, 'scale', None)
        #
        # def _freq_band_to_integer_indices(self, index):
        #     if not isinstance(index, frequencyscale.FrequencyBand):
        #         return index
        #
        #     return self.scale.get_slice(index)
        #
        # def __getitem__(self, index):
        #     try:
        #         slices = map(self._freq_band_to_integer_indices, index)
        #     except TypeError:
        #         slices = self._freq_band_to_integer_indices(index)
        #     return super(TimeFrequencyRepresentation, self).__getitem__(slices)


class TimeFrequencyRepresentationMetaData(ConstantRateTimeSeriesMetadata):
    def __init__(
            self,
            dtype=None,
            shape=None,
            frequency=None,
            duration=None,
            scale=None):
        super(TimeFrequencyRepresentationMetaData, self).__init__(
                dtype=dtype,
                shape=shape,
                frequency=frequency,
                duration=duration)
        self.scale = self._decode_scale(scale)

    @staticmethod
    def from_time_frequency_representation(tf):
        return TimeFrequencyRepresentationMetaData(
                dtype=tf.dtype,
                shape=tf.shape[1:],
                frequency=tf.frequency,
                duration=tf.duration,
                scale=tf.scale)

    def _encode_scale(self, scale):
        return \
            scale.__class__.__name__, \
            (scale.start_hz, scale.stop_hz), \
            scale.n_bands

    def _decode_scale(self, t):
        if isinstance(t, frequencyscale.FrequencyScale):
            return t
        cls_name, span, n_bands = t
        scale_cls = getattr(frequencyscale, cls_name)
        return scale_cls(frequencyscale.FrequencyBand(*span), n_bands)

    def __repr__(self):
        return repr((
            str(np.dtype(self.dtype)),
            self.shape,
            self._encode_timedelta(self.frequency),
            self._encode_timedelta(self.duration),
            self._encode_scale(self.scale)
        ))


class TimeFrequencyRepresentationEncoder(ff.NumpyEncoder):
    def __init__(self, needs=None):
        super(TimeFrequencyRepresentationEncoder, self).__init__(needs=needs)

    def _prepare_data(self, data):
        return data

    def _prepare_metadata(self, data):
        return TimeFrequencyRepresentationMetaData \
            .from_time_frequency_representation(data)


class TimeFrequencyRepresentationDecoder(ff.BaseNumpyDecoder):
    def __init__(self):
        super(TimeFrequencyRepresentationDecoder, self).__init__()

    def _unpack_metadata(self, flo):
        return TimeFrequencyRepresentationMetaData.unpack(flo)

    def _wrap_array(self, raw, metadata):
        return TimeFrequencyRepresentation(
                raw, metadata.frequency, metadata.duration, metadata.scale)


class TimeFrequencyRepresentationFeature(ff.Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=TimeFrequencyRepresentationEncoder,
            decoder=TimeFrequencyRepresentationDecoder(),
            **extractor_args):
        super(TimeFrequencyRepresentationFeature, self).__init__(
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
