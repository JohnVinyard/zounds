from featureflow import Node, NotEnoughData
import marshal
import types
from collections import OrderedDict
import hashlib
import numpy as np


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
        return data


class UnitNorm(Preprocessor):
    def __init__(self, needs=None):
        super(UnitNorm, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            from zounds.nputil import safe_unit_norm
            from zounds.core import ArrayWithUnits
            normed = safe_unit_norm(d.reshape((d.shape[0], -1)))
            normed = normed.reshape(d.shape)
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
            import numpy as np
            return np.sign(d) * np.log(np.abs(d) + 1)

        return x

    def _backward_func(self):
        def x(d):
            import numpy as np
            return (np.exp(np.abs(d)) - 1) * np.sign(d)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Log')


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
            print fill_func
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


class Flatten(Preprocessor):
    def __init__(self, needs=None):
        super(Flatten, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            return d.reshape((d.shape[0], -1))

        return x

    def _inversion_data(self):
        def x(d):
            return dict(shape=d.shape)

        return x

    def _backward_func(self):
        def x(d, shape=None):
            return d.reshape(shape)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Flatten')


class ExpandDims(Preprocessor):
    def __init__(self, needs=None):
        super(ExpandDims, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            return d[..., None]

        return x

    def _backward_func(self):
        def x(d):
            return d.reshape(d.shape[:-1])

        return x

    def _process(self, data):
        data = self._extract_data(data)
        op = self.transform()
        inv_data = self.inversion_data()
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='ExpandDims')


class MeanStdNormalization(Preprocessor):
    def __init__(self, needs=None):
        super(MeanStdNormalization, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d, mean=None, std=None):
            return (d - mean) / std

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
    def __init__(self, needs=None):
        super(InstanceScaling, self).__init__(needs=needs)

    def _forward_func(self):
        def x(d):
            import numpy as np
            print d
            print np.__version__
            axes = tuple(range(1, len(d.shape)))
            return d / np.max(np.abs(d), axis=axes, keepdims=True)

        return x

    def _inversion_data(self):
        def x(d):
            import numpy as np
            axes = tuple(range(1, len(d.shape)))
            return dict(max=np.max(np.abs(d), axis=axes, keepdims=True))

        return x

    def _backward_func(self):
        def x(d, max=None):
            return d * max

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


class Binarize(Preprocessor):
    def __init__(self, threshold=0.5, needs=None):
        super(Binarize, self).__init__(needs=needs)
        self.threshold = threshold

    def _forward_func(self):
        def x(d, threshold=None):
            import numpy as np
            data = np.zeros(d.shape, dtype=np.uint8)
            data[np.where(d > threshold)] = 1
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
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='Binarize')


class Pipeline(object):
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

    def transform(self, data):
        inversion_data = []
        wrap_data = []
        for p in self.processors:
            inversion_data.append(p.inversion_data(data))
            wrap_data.append(self.wrap_data(data))
            data = p.op(data)
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
    def __init__(self, needs=None):
        super(PreprocessingPipeline, self).__init__(needs=needs)
        self._pipeline = OrderedDict((id(n), None) for n in needs)

    def _enqueue(self, data, pusher):
        self._pipeline[id(pusher)] = data

    def _dequeue(self):
        # TODO: Make this an aggregator
        if not self._finalized or not all(self._pipeline.itervalues()):
            raise NotEnoughData()

        return Pipeline(map(
            lambda x: x.for_storage(),
            self._pipeline.itervalues()))
