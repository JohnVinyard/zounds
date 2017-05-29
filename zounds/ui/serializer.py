import datetime
from zounds.persistence import ArrayWithUnitsFeature
from zounds.timeseries import Seconds, Picoseconds, TimeSlice, AudioSamples
from zounds.segment import TimeSliceFeature
from zounds.index import SearchResults
from zounds.soundfile import OggVorbisFeature
from soundfile import SoundFile
from contentrange import ContentRange
import numpy as np
from featureflow import Decoder
import matplotlib
import json
import base64

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO


class TempResult(object):
    def __init__(
            self,
            data,
            content_type,
            is_partial=False,
            content_range=None):
        self.content_range = content_range
        self.data = data
        self.content_type = content_type
        self.timestamp = datetime.datetime.utcnow()
        self.is_partial = is_partial


class DefaultSerializer(object):
    def __init__(self, content_type):
        super(DefaultSerializer, self).__init__()
        self._content_type = content_type

    def matches(self, context):
        if context.feature is None:
            return False
        return context.feature.encoder.content_type == self._content_type

    @property
    def content_type(self):
        return self._content_type

    def serialize(self, context):
        document = context.document
        feature = context.feature
        slce = context.slce
        flo = feature(_id=document._id, decoder=Decoder(), persistence=document)
        if slce.start:
            flo.seek(slce.start)
        if slce.stop:
            value = flo.read(slce.stop - slce.start)
        else:
            value = flo.read()
        key = document.key_builder.build(
                document._id, feature.key, feature.version)
        total = document.database.size(key)
        return TempResult(
                value,
                self.content_type,
                is_partial=slce.start is not None or slce.stop is not None,
                content_range=ContentRange.from_slice(slce, total))


class AudioSamplesSerializer(object):
    def __init__(self):
        super(AudioSamplesSerializer, self).__init__()

    def matches(self, context):
        return isinstance(context.value, AudioSamples)

    @property
    def content_type(self):
        return 'audio/ogg'

    def serialize(self, context):
        bio = BytesIO()
        samples = context.value
        with SoundFile(
                bio,
                mode='w',
                samplerate=samples.samples_per_second,
                channels=samples.channels,
                format='OGG',
                subtype='VORBIS') as sf:
            for i in xrange(0, len(samples), samples.samples_per_second):
                sf.write(samples[i: i + samples.samples_per_second])
        bio.seek(0)
        return TempResult(bio.read(), 'audio/ogg')


class OggVorbisSerializer(object):
    """
    Serializer capable of handling range requests against ogg vorbis files
    """

    def __init__(self):
        super(OggVorbisSerializer, self).__init__()

    def matches(self, context):
        if context.feature is None:
            return False
        return \
            isinstance(context.feature, OggVorbisFeature) \
            and isinstance(context.slce, TimeSlice)

    @property
    def content_type(self):
        return 'audio/ogg'

    def serialize(self, context):
        feature = context.feature
        document = context.document
        slce = context.slce
        wrapper = feature(_id=document._id, persistence=document)
        samples = wrapper[slce]
        bio = BytesIO()
        with SoundFile(
                bio,
                mode='w',
                samplerate=wrapper.samplerate,
                channels=wrapper.channels,
                format='OGG',
                subtype='VORBIS') as sf:
            sf.write(samples)
        bio.seek(0)
        content_range = ContentRange.from_timeslice(
                slce, Picoseconds(int(1e12 * wrapper.duration_seconds)))
        return TempResult(
                bio.read(),
                'audio/ogg',
                is_partial=slce != TimeSlice(),
                content_range=content_range)


def generate_image(data, is_partial=False, content_range=None):
    fig = plt.figure()
    if len(data.shape) == 1:
        plt.plot(data)
    elif len(data.shape) == 2:
        data = np.abs(np.asarray(data))
        mat = plt.matshow(np.rot90(data), cmap=plt.cm.viridis)
        mat.axes.get_xaxis().set_visible(False)
        mat.axes.get_yaxis().set_visible(False)
    else:
        raise ValueError('cannot handle dimensions > 2')
    bio = BytesIO()
    plt.savefig(bio, bbox_inches='tight', pad_inches=0, format='png')
    bio.seek(0)
    fig.clf()
    plt.close('all')
    return TempResult(
            bio.read(),
            'image/png',
            is_partial=is_partial,
            content_range=content_range)


class ConstantRateTimeSeriesSerializer(object):
    def __init__(self):
        super(ConstantRateTimeSeriesSerializer, self).__init__()

    def matches(self, context):
        return \
            isinstance(context.feature, ArrayWithUnitsFeature) \
            and isinstance(context.slce, TimeSlice)

    @property
    def content_type(self):
        return 'image/png'

    def serialize(self, context):
        feature = context.feature
        document = context.document
        data = feature(_id=document._id, persistence=document)
        sliced_data = data[context.slce]
        td = data.dimensions[0]
        content_range = ContentRange.from_timeslice(
                context.slce, td.end)
        return generate_image(
                sliced_data,
                is_partial=True,
                content_range=content_range)


class NumpySerializer(object):
    def __init__(self):
        super(NumpySerializer, self).__init__()

    def matches(self, context):
        return \
            isinstance(context.value, np.ndarray) \
            and len(context.value.shape) in (1, 2)

    @property
    def content_type(self):
        return 'image/png'

    def serialize(self, context):
        feature = context.feature
        document = context.document
        value = context.value
        if value is None:
            data = feature(_id=document._id, persistence=document)
        else:
            data = value
        return generate_image(data)


class AudioSliceSerializer(object):
    def __init__(
            self,
            content_type,
            visualization_feature,
            audio_feature,
            path_builder):
        self.audio_feature = audio_feature
        self.visualization_feature = visualization_feature
        self._content_type = content_type
        self._path_builder = path_builder

    @property
    def content_type(self):
        return self._content_type

    def _seconds(self, ts):
        return {
            'start': ts.start / Seconds(1),
            'duration': ts.duration / Seconds(1)
        }

    def _result(self, ts, _id):
        return {
            'audio': self._path_builder(_id, self.audio_feature.key),
            'visualization': self._path_builder(
                    _id, self.visualization_feature.key),
            'slice': self._seconds(ts)
        }

    def iter_results(self, context):
        raise NotImplementedError()

    def additional_data(self, context):
        return dict()

    def serialize(self, context):
        results = map(lambda x: self._result(*x), self.iter_results(context))
        output = {'results': results}
        output.update(self.additional_data(context))
        return TempResult(json.dumps(output), self.content_type)


class OnsetsSerializer(AudioSliceSerializer):
    def __init__(self, visualization_feature, audio_feature, path_builder):
        super(OnsetsSerializer, self).__init__(
                'application/vnd.zounds.onsets+json',
                visualization_feature,
                audio_feature,
                path_builder)

    def matches(self, context):
        return isinstance(context.feature, TimeSliceFeature)

    def iter_results(self, context):
        for ts in context.value:
            yield ts, context.document._id


class SearchResultsSerializer(AudioSliceSerializer):
    def __init__(self, visualization_feature, audio_feature, path_builder):
        super(SearchResultsSerializer, self).__init__(
                'application/vnd.zounds.searchresults+json',
                visualization_feature,
                audio_feature,
                path_builder)

    def matches(self, context):
        return isinstance(context.value, SearchResults)

    def additional_data(self, context):
        return {'query': base64.b64encode(context.value.query)}

    def iter_results(self, context):
        for _id, ts in context.value:
            yield ts, _id
