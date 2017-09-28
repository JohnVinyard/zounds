import numpy as np
import frequencyscale
from zounds.core import Dimension
from zounds.spectral.frequencyscale import ExplicitScale


class FrequencyDimension(Dimension):
    """
    When applied to an axis of :class:`~zounds.core.ArrayWithUnits`, that axis
    can be viewed as representing the energy present in a series of frequency
    bands

    Args:
        scale (FrequencyScale): A scale whose frequency bands correspond to the
            items along the frequency axis

    Examples:
        >>> from zounds import LinearScale, FrequencyBand, ArrayWithUnits
        >>> from zounds import FrequencyDimension
        >>> import numpy as np
        >>> band = FrequencyBand(20, 20000)
        >>> scale = LinearScale(frequency_band=band, n_bands=100)
        >>> raw = np.hanning(100)
        >>> arr = ArrayWithUnits(raw, [FrequencyDimension(scale)])
        >>> sliced = arr[FrequencyBand(100, 1000)]
        >>> sliced.shape
        (5,)
        >>> sliced.dimensions
        (FrequencyDimension(scale=LinearScale(band=FrequencyBand(
        start_hz=20.0,
        stop_hz=1019.0,
        center=519.5,
        bandwidth=999.0), n_bands=5)),)
    """
    def __init__(self, scale):
        super(FrequencyDimension, self).__init__()
        self.scale = scale

    def weights(self, weights, arr, i):
        return weights

    def modified_dimension(self, size, windowsize, stepsize=None):
        raise NotImplementedError()

    def metaslice(self, index, size):
        return FrequencyDimension(self.scale[index])

    def integer_based_slice(self, index):
        if not isinstance(index, frequencyscale.FrequencyBand):
            return index

        return self.scale.get_slice(index)

    def validate(self, size):
        """
        Ensure that the size of the dimension matches the number of bands in the
        scale

        Raises:
             ValueError: when the dimension size and number of bands don't match
        """
        msg = 'scale and array size must match, ' \
              'but were scale: {self.scale.n_bands},  array size: {size}'

        if size != len(self.scale):
            raise ValueError(msg.format(**locals()))

    def __eq__(self, other):
        return self.scale == other.scale

    def __str__(self):
        return 'FrequencyDimension(scale={self.scale})'.format(**locals())

    def __repr__(self):
        return self.__str__()


class ExplicitFrequencyDimension(Dimension):
    """
    A frequency dimension where the mapping from frequency bands to integer
    indices is provided explicitly, rather than computed

    Args:
        scale (ExplicitScale): the explicit frequency scale that defines how
            slices are extracted from this dimension
        slices (iterable of slices): An iterable of :class:`python.slice`
            instances which correspond to each frequency band from scale

    Raises:
        ValueError: when the number of slices and number of bands in scale don't
            match
    """

    def __init__(self, scale, slices):
        super(ExplicitFrequencyDimension, self).__init__()
        if len(scale) != len(slices):
            raise ValueError('scale and slices must have same length')

        self.scale = scale
        self.slices = slices
        self._lookup = dict(zip(self.scale, self.slices))

    def weights(self, weights, arr, i):
        w = np.zeros(self.slices[-1].stop - self.slices[0].start)
        for weight, sl in zip(weights, self.slices):
            w[sl] = weight
        return w

    def modified_dimension(self, size, windowsize, stepsize=None):
        raise NotImplementedError()

    def metaslice(self, index, size):
        if isinstance(index, frequencyscale.FrequencyBand):
            try:
                slce = self._lookup[index]
                return ExplicitFrequencyDimension(
                    ExplicitScale([index]), [slce])
            except KeyError:
                slce = self.scale.get_slice(index)
        else:
            slce = index

        print slce
        return ExplicitFrequencyDimension(self.scale[slce], self.slices[slce])

    def integer_based_slice(self, index):
        if not isinstance(index, frequencyscale.FrequencyBand):
            return index

        try:
            return self._lookup[index]
        except KeyError:
            pass

        slce = self.scale.get_slice(index)
        slices = self.slices[slce]
        return slice(slices[0].start, slices[-1].stop)

    def validate(self, size):
        return size == self.slices[-1].stop

    def __eq__(self, other):
        return self.scale == other.scale and self.slices == other.slices

    def __str__(self):
        return \
            'ExplicitFrequencyDimension(scale={self.scale}, slices=self.slices)' \
                .format(**locals())

    def __repr__(self):
        return self.__str__()
