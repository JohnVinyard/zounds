import os
import sys
import json
import tornado.ioloop
import tornado.web
import httplib
import numpy as np
from featureflow import Decoder
import traceback
import urllib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
from cStringIO import StringIO
import ast
import re
import uuid
import datetime
from zounds.timeseries import \
    ConstantRateTimeSeriesFeature, AudioSamples, Seconds, Picoseconds, TimeSlice
from zounds.segment import TimeSliceFeature
from zounds.index import SearchResults
from zounds.soundfile import OggVorbisFeature
from soundfile import SoundFile


class RangeUnitUnsupportedException(Exception):
    """
    Raised when an HTTP range request is made with a unit not supported by this
    application
    """
    pass


class UnsatisfiableRangeRequestException(Exception):
    """
    Exception raised when an HTTP range request cannot be satisfied for a
    particular resource
    """
    pass


class RangeRequest(object):
    def __init__(self, range_header):
        self.range_header = range_header
        self.re = re.compile(
                r'^(?P<unit>[^=]+)=(?P<start>[^-]+)-(?P<stop>.*?)$')

    def time_slice(self, start, stop):
        start = float(start)
        try:
            stop = float(stop)
        except ValueError:
            stop = None
        duration = \
            None if stop is None else Picoseconds(int(1e12 * (stop - start)))
        start = Picoseconds(int(1e12 * start))
        return TimeSlice(duration, start=start)

    def byte_slice(self, start, stop):
        start = int(start)
        try:
            stop = int(stop)
        except ValueError:
            stop = None
        return slice(start, stop)

    def range(self):
        raw = self.range_header
        if not raw:
            return slice(None)

        m = self.re.match(raw)
        if not m:
            return slice(None)

        units = m.groupdict()['unit']
        start = m.groupdict()['start']
        stop = m.groupdict()['stop']

        if units == 'bytes':
            return self.byte_slice(start, stop)
        elif units == 'seconds':
            return self.time_slice(start, stop)
        else:
            raise RangeUnitUnsupportedException(units)


class NoMatchingSerializerException(Exception):
    pass


class ContentRange(object):
    def __init__(self, unit, start, total, stop=None):
        self.unit = unit
        self.total = total
        self.stop = stop
        self.start = start

    @staticmethod
    def from_timeslice(timeslice, total):
        one_second = Seconds(1)
        stop = None
        start = timeslice.start / one_second
        if timeslice.duration is not None:
            stop = start + (timeslice.duration / one_second)
        return ContentRange(
                'seconds',
                timeslice.start / one_second,
                total / one_second,
                stop)

    @staticmethod
    def from_slice(slce, total):
        return ContentRange('bytes', slce.start, total, slce.stop)

    def __str__(self):
        unit = self.unit
        start = self.start
        stop = self.stop or self.total
        total = self.total
        return '{unit} {start}-{stop}/{total}'.format(**locals())


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


class RequestContext(object):
    def __init__(
            self,
            document=None,
            feature=None,
            slce=None,
            value=None):
        self.value = value
        self.slce = slce
        self.feature = feature
        self.document = document

    def __repr__(self):
        return '''RequestContext(
    document={document},
    feature={feature},
    slce={slce},
    value={value})'''.format(**self.__dict__)

    def __str__(self):
        return self.__repr__()


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
        mat = plt.matshow(np.rot90(data), cmap=plt.cm.gray)
        mat.axes.get_xaxis().set_visible(False)
        mat.axes.get_yaxis().set_visible(False)
    else:
        raise ValueError('cannot handle dimensions > 2')
    bio = BytesIO()
    plt.savefig(bio, bbox_inches='tight', pad_inches=0, format='png')
    bio.seek(0)
    fig.clf()
    plt.close()
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
            isinstance(context.feature, ConstantRateTimeSeriesFeature) \
            and isinstance(context.slce, TimeSlice)

    @property
    def content_type(self):
        return 'image/png'

    def serialize(self, context):
        feature = context.feature
        document = context.document
        data = feature(_id=document._id, persistence=document)
        sliced_data = data[context.slce]
        content_range = ContentRange.from_timeslice(
                context.slce, data.end)
        return generate_image(
                sliced_data,
                is_partial=True,
                content_range=content_range)


class NumpySerializer(object):
    def __init__(self):
        super(NumpySerializer, self).__init__()

    def matches(self, context):
        if context.document is not None \
                and isinstance(context.feature, ConstantRateTimeSeriesFeature):
            return True

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

    def serialize(self, context):
        results = map(lambda x: self._result(*x), self.iter_results(context))
        output = {'results': results}
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

    def iter_results(self, context):
        for _id, ts in context.value:
            yield ts, _id


class ZoundsApp(object):
    def __init__(
            self,
            base_path=r'/zounds/',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            globals={},
            locals={}):

        super(ZoundsApp, self).__init__()
        self.locals = locals
        self.globals = globals
        self.model = model
        self.visualization_feature = visualization_feature
        self.audio_feature = audio_feature
        self.base_path = base_path
        self.serializers = [
            AudioSamplesSerializer(),
            OggVorbisSerializer(),
            ConstantRateTimeSeriesSerializer(),
            DefaultSerializer('application/json'),
            DefaultSerializer('audio/ogg'),
            NumpySerializer(),
            OnsetsSerializer(
                    self.visualization_feature,
                    self.audio_feature,
                    self.feature_path),
            SearchResultsSerializer(
                    self.visualization_feature,
                    self.audio_feature,
                    self.feature_path)
        ]
        self.temp = {}

        path, fn = os.path.split(__file__)
        with open(os.path.join(path, 'zounds.js')) as f:
            script = f.read()

        with open(os.path.join(path, 'index.html')) as f:
            self._html_content = f.read().replace(
                    '<script src="/zounds.js"></script>',
                    '<script>{}</script>'.format(script))

    def feature_path(self, _id, feature):
        _id = urllib.quote(_id, safe='')
        return '{base_path}{_id}/{feature}'.format(
                base_path=self.base_path, _id=_id, feature=feature)

    def find_serializer(self, context):
        try:
            return filter(
                    lambda x: x.matches(context), self.serializers)[0]
        except IndexError:
            raise NoMatchingSerializerException()

    def serialize(self, context):
        serializer = self.find_serializer(context)
        return serializer.serialize(context)

    def temp_handler(self):

        app = self

        class TempHandler(tornado.web.RequestHandler):

            def get(self, _id):
                try:
                    result = app.temp[_id]
                except KeyError:
                    self.set_status(httplib.NOT_FOUND)
                    self.finish()
                    return
                self.set_header('Content-Type', result.content_type)
                self.set_header('Accept-Ranges', 'bytes')
                self.write(result.data)
                self.set_status(httplib.OK)
                self.finish()

        return TempHandler

    def repl_handler(self):
        document = self.model
        globals = self.globals
        locals = self.locals
        app = self

        class ReplHandler(tornado.web.RequestHandler):

            def _extract_feature(self, statement):
                root = ast.parse(statement)
                nodes = list(ast.walk(root))
                doc = None
                feature = None
                feature_name = None

                for node in nodes:
                    if doc and feature_name:
                        break

                    if isinstance(node, ast.Name) \
                            and node.id in locals \
                            and isinstance(locals[node.id], document):
                        doc = locals[node.id]
                        continue

                    if isinstance(node, ast.Attribute) \
                            and node.attr in document.features:
                        feature_name = node.attr
                        continue

                if feature_name:
                    feature = document.features[feature_name]

                return doc, feature

            def _add_url(self, statement, output, value):
                doc, feature = self._extract_feature(statement)
                try:
                    context = RequestContext(
                            document=doc,
                            feature=feature,
                            value=value,
                            slce=slice(None))
                    result = app.serialize(context)
                    temp_id = uuid.uuid4().hex
                    app.temp[temp_id] = result
                    output['url'] = '/zounds/temp/{temp_id}'.format(
                            temp_id=temp_id)
                    output['contentType'] = result.content_type
                except NoMatchingSerializerException:
                    pass

            def post(self):
                statement = self.request.body
                self.set_header('Content-Type', 'application/json')
                output = dict()

                try:
                    try:
                        value = eval(statement, globals, locals)
                        output['result'] = str(value)
                        self._add_url(statement, output, value)
                    except SyntaxError:
                        orig_stdout = sys.stdout
                        sys.stdout = sio = StringIO()
                        exec (statement, globals, locals)
                        sio.seek(0)
                        output['result'] = sio.read()
                        sys.stdout = orig_stdout
                    self.set_status(httplib.OK)
                except:
                    output['error'] = traceback.format_exc()
                    self.set_status(httplib.BAD_REQUEST)

                self.write(json.dumps(output))
                self.finish()

        return ReplHandler

    def feature_handler(self):

        document = self.model
        app = self

        class FeatureHandler(tornado.web.RequestHandler):

            def get(self, _id, feature):
                doc = document(_id)
                feature = document.features[feature]
                try:
                    slce = RangeRequest(self.request.headers['Range']).range()
                except KeyError:
                    slce = slice(None)
                context = RequestContext(
                        document=doc, feature=feature, slce=slce)
                try:
                    result = app.serialize(context)
                except UnsatisfiableRangeRequestException:
                    self.set_status(httplib.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.finish()
                self.set_header('Content-Type', result.content_type)
                self.set_header('Accept-Ranges', 'bytes')
                self.write(result.data)
                self.set_header('ETag', self.compute_etag())
                self.set_status(
                        httplib.PARTIAL_CONTENT if result.is_partial
                        else httplib.OK)
                if result.content_range:
                    self.set_header('Content-Range', str(result.content_range))
                self.finish()

        return FeatureHandler

    def ui_handler(self):
        app = self

        class UIHandler(tornado.web.RequestHandler):
            def get(self):
                self.set_header('Content-Type', 'text-html')
                self.write(app._html_content)
                self.set_status(httplib.OK)
                self.finish()

        return UIHandler

    def build_app(self):
        return tornado.web.Application([
            (r'/', self.ui_handler()),
            (r'/zounds/temp/(.+?)/?', self.temp_handler()),
            (r'/zounds/repl/?', self.repl_handler()),
            (r'/zounds/(.+?)/(.+?)/?', self.feature_handler())
        ])

    def start(self, port=8888):
        app = self.build_app()
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()
