import numpy as np


# TODO: Factor this common behavior from WindowingFunc out into a common
# location
class FrequencyWeighting(object):
    def __init__(self):
        super(FrequencyWeighting, self).__init__()

    def __numpy_ufunc__(self, *args, **kwargs):
        raise NotImplementedError()

    def _wdata(self, other):
        return np.ones(other.shape)

    def __mul__(self, other):
        try:
            return self._wdata(other) * other
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

    def _wdata(self, other):
        center_frequencies = np.array(list(other.scale.center_frequencies)) ** 2
        a = (12200 ** 2) * (center_frequencies ** 2)
        b = center_frequencies + (20.6 ** 2)
        c = center_frequencies + (107.7 ** 2)
        d = center_frequencies + (737.9 ** 2)
        e = center_frequencies + (12200 ** 2)
        f = a / (b * np.sqrt(c * d) * e)
        result = 2.0 + (20 * np.log10(f))
        return 1 + (result - np.min(result))
