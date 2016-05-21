from featureflow import Node, NotEnoughData, Aggregator
import numpy as np
from collections import OrderedDict
from zounds.timeseries import ConstantRateTimeSeries, VariableRateTimeSeries


class Merge(Node):
    """
    Combine two or more sources into a single feature
    """

    def __init__(self, needs=None):
        super(Merge, self).__init__(needs=needs)
        exc_msg = 'you must supply two or more dependencies'
        try:
            needs = list(needs)
        except TypeError:
            raise ValueError(exc_msg)

        if len(needs) < 2:
            raise ValueError(exc_msg)

        self._cache = OrderedDict((id(n), None) for n in needs)

    def _enqueue(self, data, pusher):
        key = id(pusher)
        if self._cache[key] is None or self._cache[key].size == 0:
            self._cache[key] = data
        else:
            self._cache[key] = self._cache[key].concatenate(data)

    def _dequeue(self):
        if any(v is None or len(v) == 0 for v in self._cache.itervalues()):
            raise NotEnoughData()
        shortest = min(len(v) for v in self._cache.itervalues())
        output = OrderedDict(
                (k, v[:shortest]) for k, v in self._cache.iteritems())
        self._cache = OrderedDict(
                (k, v[shortest:]) for k, v in self._cache.iteritems())
        return output

    def _process(self, data):
        yield ConstantRateTimeSeries.concat(data.values(), axis=1)


# KLUDGE: This implementation may currently only be used when consuming
# from a stored SparseTimeSliceFeature.  If it is run during initial computation,
# it will receive only timestamps, and the logic will break.  This is a bug
# in the BasePeakPicker class that needs to be fixed.  It should emit
# timeslices, and not timestamps
class Pooled(Aggregator, Node):
    def __init__(self, op=None, axis=None, needs=None):
        super(Pooled, self).__init__(needs=needs)
        self._timeslices = []
        self._timeseries = None
        self._op = op
        self._axis = axis

    def _enqueue(self, data, pusher):
        if isinstance(data, ConstantRateTimeSeries):
            try:
                self._timeseries.concatenate(data)
            except AttributeError:
                self._timeseries = data
        else:
            self._timeslices.extend(data)

    def _dequeue(self):
        examples = [(ts, self._op(self._timeseries[ts], axis=self._axis))
                for ts in self._timeslices]
        return VariableRateTimeSeries(examples)


class Slice(Node):
    def __init__(self, sl=None, needs=None):
        super(Slice, self).__init__(needs=needs)
        self._sl = sl

    def _process(self, data):
        yield data[:, self._sl]


class Sum(Node):
    def __init__(self, axis=0, needs=None):
        super(Sum, self).__init__(needs=needs)
        self._axis = axis

    def _process(self, data):
        # TODO: This should be generalized.  Sum will have this same problem
        try:
            data = np.sum(data, axis=self._axis)
        except ValueError:
            print 'ERROR'
            data = data
        if data.shape[0]:
            yield data


class Max(Node):
    def __init__(self, axis=0, needs=None):
        super(Max, self).__init__(needs=needs)
        self._axis = axis

    def _process(self, data):
        # TODO: This should be generalized.  Sum will have this same problem
        try:
            data = np.max(data, axis=self._axis)
        except ValueError:
            print 'ERROR'
            data = data
        if data.shape[0]:
            yield data
