from featureflow import Node, NotEnoughData
import marshal
import types
from collections import OrderedDict
import hashlib
import numpy as np
from functional import hyperplanes


class Op(object):
    def __init__(self, func, **kwargs):
        super(Op, self).__init__()
        self._kwargs = kwargs
        try:
            self._func = marshal.dumps(func.func_code)
        except AttributeError:
            # func is already a marshalled function
            self._func = func
        self._version = self._compute_version()

    @property
    def version(self):
        return self._version

    @property
    def kwargs(self):
        return self._kwargs

    def _compute_version(self):
        h = hashlib.md5(self._func)
        for v in self._kwargs.itervalues():
            try:
                h.update(v.version)
            except AttributeError:
                h.update(np.array(v))
        return h.hexdigest()

    def __getattr__(self, key):
        if key == '_kwargs':
            raise AttributeError()

        try:
            return self._kwargs[key]
        except KeyError:
            raise AttributeError(key)

    def __call__(self, arg, **kwargs):
        code = marshal.loads(self._func)
        f = types.FunctionType(
            code,
            globals(),
            'preprocess')
        kwargs.update(self._kwargs)
        return f(arg, **kwargs)


class PreprocessResult(object):
    """
    `PreprocessResult` are the output of :class:`Preprocessor` nodes, and can
    participate in a `Pipeline`.

    Args:
        data: the data on which the node in the graph was originally trained
        op (Op): a callable that can transform data
        inversion_data: data extracted in the forward pass of the model, that
            can be used to invert the result
        inverse (Op): a callable that given the output of `op`, and
            `inversion_data`, can invert the result
    """

    def __init__(self, data, op, inversion_data=None, inverse=None, name=None):
        super(PreprocessResult, self).__init__()
        self.name = name
        self.inverse = inverse
        self.inversion_data = inversion_data
        self.data = data
        self.op = op

    def __str__(self):
        return 'PreprocessResult(name={name})'.format(**self.__dict__)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, key):
        if key == 'op':
            raise AttributeError()
        return getattr(self.op, key)

    def for_storage(self):
        return PreprocessResult(
            None,
            self.op,
            self.inversion_data,
            self.inverse,
            self.name)


class Preprocessor(Node):
    """
    `Preprocessor` is the common base class for nodes in a processing graph that
    will produce :class:`PreprocessingResult` instances that end up as part of
    a :class:`Pipeline`.

    Args:
        needs (Node): previous processing node(s) on which this one depends
            for its data

    See Also:
        :class:`PreprocessResult`
        :class:`PreprocessingPipeline`
        :class:`PipelineResult`
    """

    def __init__(self, needs=None):
        super(Preprocessor, self).__init__(needs=needs)

    def _forward_func(self):
        """
        Return a function that represents this processor's
        forward transform
        """

        def x(data, example_arg=None, another_one=None):
            return data

        return x

    def transform(self, **kwargs):
        return Op(self._forward_func(), **kwargs)

    def _inversion_data(self):
        """
        Return a function that computes any data needed for
        an inverse transform, if one is possible.  Otherwise,
        raise NotImplemented
        """

        def x(data, **kwargs):
            return kwargs

        return x

    def inversion_data(self, **kwargs):
        return Op(self._inversion_data(), **kwargs)

    def _backward_func(self):
        """
        Return a function that computes this processor's
        inverse transform, if one is possible.  Otherwise,
        raise NotImplemented
        """

        def x(data, **inversion_args):
            return data

        return x

    def inverse_transform(self):
        return Op(self._backward_func())

    def _extract_data(self, data):
        if isinstance(data, PreprocessResult):
            return data.data
        elif isinstance(data, dict):
            return dict((k, self._extract_data(v)) for k, v in data.iteritems())
        else:
            return data


class UnitNorm(Preprocessor):
    def __init__(self, needs=None):
        super(UnitNorm, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            from zounds.core import ArrayWithUnits
            from functional import example_wise_unit_norm
            normed = example_wise_unit_norm(d)
            try:
                return ArrayWithUnits(normed, d.dimensions)
            except AttributeError:
                return normed

        return x

    def _inversion_data(self):
        def x(d):
            import numpy as np
            return dict(norm=np.linalg.norm(
                d.reshape((d.shape[0], -1)), axis=1))

        return x

    def _backward_func(self):
        def x(d, norm=None):
            return (d.T * norm).T

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='UnitNorm')


class Log(Preprocessor):
    """
    Perform the log-modulus transform on data
    (http://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html)

    This transform will tend to compress the overall range of values
    """

    def __init__(self, needs=None):
        super(Log, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            from zounds.loudness import log_modulus
            return log_modulus(d)

        return x

    def _backward_func(self):
        def x(d):
            from zounds.loudness import inverse_log_modulus
            return inverse_log_modulus(d)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Log')


class MuLawCompressed(Preprocessor):
    def __init__(self, needs=None):
        super(MuLawCompressed, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            from zounds.loudness import mu_law
            return mu_law(d)

        return x

    def _backward_func(self):
        def x(d):
            from zounds.loudness import inverse_mu_law
            return inverse_mu_law(d)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='MuLawCompressed')


class Slicer(Preprocessor):
    def __init__(self, slicex=None, needs=None):
        super(Slicer, self).__init__(needs=needs)
        self.fill_func = np.zeros
        self.slicex = slicex

    def _forward_func(self):
        def x(d, slicex=None):
            return d[..., slicex]

        return x

    def _inversion_data(self):
        def y(d, slicex=None, fill_func=None):

            try:
                ka = d.kwargs()

                def ff(shape, dtype):
                    return d.__class__(fill_func(shape, dtype), **ka)
            except AttributeError:

                def ff(shape, dtype):
                    return fill_func(shape, dtype)

            return dict(shape=d.shape, slicex=slicex, fill_func=ff)

        return y

    def _backward_func(self):
        def z(d, shape=None, fill_func=None, slicex=None):
            new_shape = d.shape[:1] + shape[1:]
            new_arr = fill_func(new_shape, d.dtype)
            new_arr[..., slicex] = d
            return new_arr

        return z

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(slicex=self.slicex)
        inv_data = self.inversion_data(
            fill_func=self.fill_func, slicex=self.slicex)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Slicer')


class Reshape(Preprocessor):
    def __init__(self, new_shape, needs=None):
        super(Reshape, self).__init__(needs=needs)
        self.new_shape = new_shape

    def _forward_func(self):
        def x(d, new_shape=None):
            return d.reshape(d.shape[:1] + new_shape)

        return x

    def _inversion_data(self):
        def x(d):
            return dict(original_shape=d.shape)

        return x

    def _backward_func(self):
        def x(d, original_shape=None):
            return d.reshape(original_shape)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(new_shape=self.new_shape)
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Reshape')


class SimHash(Preprocessor):
    """
    Hash feature vectors by computing on which side of N hyperplanes those
    features lie.

    Args:
        bits (int): The number of hyperplanes, and hence, the number of bits
            in the resulting hash
        packbits (bool): Should the result be bit-packed?
        needs (Preprocessor): the processing node on which this node relies for
            its data
    """

    def __init__(self, bits=None, packbits=False, needs=None):
        super(SimHash, self).__init__(needs=needs)
        self.packbits = packbits
        self.bits = bits

    def _forward_func(self):
        def x(d, plane_vectors=None, packbits=None):

            from zounds.core import ArrayWithUnits, IdentityDimension
            from zounds.learn import simhash
            import numpy as np

            bits = simhash(plane_vectors, d)

            if packbits:
                bits = np.packbits(bits, axis=-1).view(np.uint64)

            try:
                return ArrayWithUnits(
                    bits, [d.dimensions[0], IdentityDimension()])
            except AttributeError:
                return bits

        return x

    def _backward_func(self):
        def x(d):
            raise NotImplementedError()

        return x

    def _process(self, data):
        data = self._extract_data(data)
        mean = data.mean(axis=0).flatten()
        std = data.std(axis=0).flatten()
        plane_vectors = hyperplanes(mean, std, self.bits)
        op = self.transform(plane_vectors=plane_vectors, packbits=self.packbits)
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='SimHash')


class MeanStdNormalization(Preprocessor):
    def __init__(self, needs=None):
        super(MeanStdNormalization, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d, mean=None, std=None):
            import numpy as np
            return np.divide(d - mean, std, where=std != 0)

        return x

    def _backward_func(self):
        def x(d, mean=None, std=None):
            arr = d.copy()
            arr *= std
            arr += mean
            return arr

        return x

    def _process(self, data):
        data = self._extract_data(data)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        op = self.transform(mean=mean, std=std)
        inv_data = self.inversion_data(mean=mean, std=std)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='MeanStd')


class InstanceScaling(Preprocessor):
    def __init__(self, max_value=1, needs=None):
        super(InstanceScaling, self).__init__(needs=needs)
        self.max_value = max_value

    def _forward_func(self):
        def x(d, max_value=None):
            import numpy as np
            axes = tuple(range(1, len(d.shape)))
            m = np.max(np.abs(d), axis=axes, keepdims=True)
            return max_value * np.divide(d, m, where=m != 0)

        return x

    def _inversion_data(self):
        def x(d, max_value=None):
            import numpy as np
            axes = tuple(range(1, len(d.shape)))
            return dict(
                max=np.max(np.abs(d), axis=axes, keepdims=True),
                max_value=max_value)

        return x

    def _backward_func(self):
        def x(d, max=None, max_value=None):
            return (d * max) / max_value

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(max_value=self.max_value)
        inv_data = self.inversion_data(max_value=self.max_value)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='InstanceScaling')


class Multiply(Preprocessor):
    def __init__(self, factor=1, needs=None):
        super(Multiply, self).__init__(needs=needs)
        self.factor = factor

    def _forward_func(self):
        def x(d, factor=None):
            return d * factor

        return x

    def _backward_func(self):
        def x(d, factor=None):
            return d * (1.0 / factor)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(factor=self.factor)
        inv_data = self.inversion_data(factor=self.factor)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Multiply')


class Weighted(Preprocessor):
    def __init__(self, weighting, needs=None):
        super(Weighted, self).__init__(needs=needs)
        self.weighting = weighting

    def _forward_func(self):
        def x(d, weighting=None):
            return d * weighting

        return x

    def _backward_func(self):
        def x(d, weighting=None):
            return d / weighting

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(weighting=self.weighting)
        inv_data = self.inversion_data(weighting=self.weighting)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Weighted')


class AbsoluteValue(Preprocessor):
    def __init__(self, needs=None):
        super(AbsoluteValue, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            import numpy as np
            from zounds.core import ArrayWithUnits
            processed = np.abs(d)
            try:
                return ArrayWithUnits(processed, d.dimensions)
            except AttributeError:
                return processed

        return x

    def _backward_func(self):
        def x(d):
            raise NotImplementedError()

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()

        try:
            data = op(data)
        except:
            data = None

        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv,
            name='AbsoluteValue')


class Sharpen(Preprocessor):
    def __init__(self, kernel=None, needs=None):
        super(Sharpen, self).__init__(needs=needs)
        self.kernel = kernel

    def _forward_func(self):
        def x(d, kernel=None):
            from scipy.signal import convolve
            from zounds.core import ArrayWithUnits
            data = convolve(d, kernel[None, ...], mode='same')
            data[data < 0] = 0
            try:
                return ArrayWithUnits(data, d.dimensions)
            except AttributeError:
                return data

        return x

    def _backward_func(self):
        def x(d):
            raise NotImplementedError()

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(kernel=self.kernel)
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        try:
            data = op(data)
        except:
            data = None
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Sharpen')


class Binarize(Preprocessor):
    def __init__(self, threshold=0.5, needs=None):
        super(Binarize, self).__init__(needs=needs)
        self.threshold = threshold

    def _forward_func(self):
        def x(d, threshold=None):
            import numpy as np
            from zounds.core import ArrayWithUnits
            data = np.zeros(d.shape, dtype=np.uint8)
            data[np.where(d > threshold)] = 1
            try:
                return ArrayWithUnits(data, d.dimensions)
            except AttributeError:
                return data

        return x

    def _backward_func(self):
        def x(d):
            raise NotImplementedError()

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform(threshold=self.threshold)
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        try:
            data = op(data)
        except:
            data = None
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Binarize')


class Pipeline(object):
    """

    """

    def __init__(self, preprocess_results):
        self.processors = list(preprocess_results)
        self.version = hashlib.md5(
            ''.join([p.op.version for p in self.processors])).hexdigest()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.processors[index]
        elif isinstance(index, list):
            return Pipeline([self.processors[i] for i in index])
        return Pipeline(self.processors[index])

    def wrap_data(self, data):
        cls = data.__class__
        try:
            kwargs = data.kwargs()
        except AttributeError:
            kwargs = None
        return cls, kwargs

    def transform(self, data, wrapper=None):
        inversion_data = []
        wrap_data = []
        for p in self.processors:
            inversion_data.append(p.inversion_data(data))
            wrap_data.append(self.wrap_data(data))
            data = p.op(data)

        if wrapper is not None:
            data = wrapper(data)

        return PipelineResult(data, self.processors, inversion_data, wrap_data)


class PipelineResult(object):
    def __init__(self, data, processors, inversion_data, wrap_data):
        super(PipelineResult, self).__init__()
        self.processors = processors[::-1]
        self.inversion_data = inversion_data[::-1]
        self.wrap_data = wrap_data[::-1]
        self.data = data

    def unwrap(self, data, wrap_data):
        cls, kwargs = wrap_data
        if kwargs is None:
            return data
        return cls(data, **kwargs)

    def inverse_transform(self):
        data = self.data
        for inv_data, wrap_data, p in \
                zip(self.inversion_data, self.wrap_data, self.processors):
            data = p.inverse(data, **inv_data)
            data = self.unwrap(data, wrap_data)
        return data


class PreprocessingPipeline(Node):
    """
    A `PreprocessingPipeline` is a node in the graph that can be connected to
    one or more :class:`Preprocessor` nodes, whose output it will assemble into
    a re-usable pipeline.

    Args:
        needs (list or tuple of Node): the :class:`Preprocessor` nodes on whose
            output this pipeline depends

    Here's an example of a learning pipeline that will first find the
    feature-wise mean and standard deviation of a dataset, and will then learn
    K-Means clusters from the dataset.  This will result in a re-usable pipeline
    that can use statistics from the original dataset to normalize new examples,
    assign them to a cluster, and finally, reconstruct them.

    .. code:: python

        import featureflow as ff
        import zounds
        from random import choice

        samplerate = zounds.SR44100()
        STFT = zounds.stft(resample_to=samplerate)


        @zounds.simple_in_memory_settings
        class Sound(STFT):
            bark = zounds.ArrayWithUnitsFeature(
                zounds.BarkBands,
                samplerate=samplerate,
                needs=STFT.fft,
                store=True)


        @zounds.simple_in_memory_settings
        class ExamplePipeline(ff.BaseModel):
            docs = ff.PickleFeature(
                ff.IteratorNode,
                needs=None)

            shuffled = ff.PickleFeature(
                zounds.ShuffledSamples,
                nsamples=100,
                needs=docs,
                store=False)

            meanstd = ff.PickleFeature(
                zounds.MeanStdNormalization,
                needs=docs,
                store=False)

            kmeans = ff.PickleFeature(
                zounds.KMeans,
                needs=meanstd,
                centroids=32)

            pipeline = ff.PickleFeature(
                zounds.PreprocessingPipeline,
                needs=(meanstd, kmeans),
                store=True)

        # apply the Sound processing graph to individual audio files
        for metadata in zounds.InternetArchive('TheR.H.SFXLibrary'):
            print 'processing {url}'.format(url=metadata.request.url)
            Sound.process(meta=metadata)

        # apply the ExamplePipeline processing graph to the entire corpus of audio
        _id = ExamplePipeline.process(docs=(snd.bark for snd in Sound))
        learned = ExamplePipeline(_id)

        snd = choice(list(Sound))
        result = learned.pipeline.transform(snd.bark)
        print result.data  # print the assigned centroids for each FFT frame
        inverted = result.inverse_transform()
        print inverted  # the reconstructed FFT frames


    See Also:
        :class:`Pipeline`
        :class:`Preprocessor`
        :class:`PreprocessResult`
        :class:`PipelineResult`
    """

    def __init__(self, needs=None):
        super(PreprocessingPipeline, self).__init__(needs=needs)
        self._pipeline = OrderedDict((id(n), None) for n in needs.values())

    def _enqueue(self, data, pusher):
        self._pipeline[id(pusher)] = data

    def _dequeue(self):
        if not self._finalized or not all(self._pipeline.itervalues()):
            raise NotEnoughData()

        return Pipeline(map(
            lambda x: x.for_storage(),
            self._pipeline.itervalues()))
