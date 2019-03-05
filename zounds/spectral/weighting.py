import numpy as np
from .frequencyadaptive import FrequencyAdaptive


class FrequencyWeighting(object):
    def __init__(self):
        super(FrequencyWeighting, self).__init__()

    def __numpy_ufunc__(self, *args, **kwargs):
        raise NotImplementedError()

    def _wdata(self, scale):
        return np.ones(len(scale))

    def weights(self, other):
        """
        Compute weights, given a scale or time-frequency representation
        :param other: A time-frequency representation, or a scale
        :return: a numpy array of weights
        """
        try:
            return self._wdata(other)
        except AttributeError:
            frequency_dim = other.dimensions[-1]
            return self._wdata(frequency_dim.scale)

    def _get_factors(self, arr):
        for i, d in enumerate(arr.dimensions):
            try:
                weights = self._wdata(d.scale)
                expanded = d.weights(weights, arr, i)
                return expanded
            except AttributeError as e:
                pass

        raise ValueError('arr must have a frequency dimension')

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if ufunc == np.multiply or ufunc == np.divide:
            if args[0] is self:
                first_arg = self._get_factors(args[1])
                second_arg = args[1]
            else:
                first_arg = args[0]
                second_arg = self._get_factors(args[0])
            return getattr(ufunc, method)(first_arg, second_arg, **kwargs)
        else:
            return NotImplemented


class AWeighting(FrequencyWeighting):
    """
    An A-weighting (https://en.wikipedia.org/wiki/A-weighting) that can be
    applied to a frequency axis via multiplication.

    Examples:
        >>> from zounds import ArrayWithUnits, GeometricScale
        >>> from zounds import FrequencyDimension, AWeighting
        >>> import numpy as np
        >>> scale = GeometricScale(20, 20000, 0.05, 10)
        >>> raw = np.ones(len(scale))
        >>> arr = ArrayWithUnits(raw, [FrequencyDimension(scale)])
        >>> arr * AWeighting()
        ArrayWithUnits([  1.        ,  18.3172567 ,  31.19918106,  40.54760374,
                47.15389876,  51.1554151 ,  52.59655479,  52.24516649,
                49.39906912,  42.05409205])
    """

    def __init__(self):
        super(AWeighting, self).__init__()

    def _wdata(self, scale):
        center_frequencies = np.array(list(scale.center_frequencies)) ** 2
        a = (12200 ** 2) * (center_frequencies ** 2)
        b = center_frequencies + (20.6 ** 2)
        c = center_frequencies + (107.7 ** 2)
        d = center_frequencies + (737.9 ** 2)
        e = center_frequencies + (12200 ** 2)
        f = a / (b * np.sqrt(c * d) * e)
        result = 2.0 + (20 * np.log10(f))
        return 1 + (result - np.min(result))
