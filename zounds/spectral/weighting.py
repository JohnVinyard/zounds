import numpy as np


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
            return self._wdata(other.scale)

    def __mul__(self, other):
        try:
            return self._wdata(other.scale) * other
        except AttributeError:
            return super(FrequencyWeighting, self).__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


class AWeighting(FrequencyWeighting):
    """
    https://en.wikipedia.org/wiki/A-weighting
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
