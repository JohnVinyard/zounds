import frequencyscale
from zounds.core import Dimension
from zounds.spectral.frequencyscale import ExplicitScale


class FrequencyDimension(Dimension):
    def __init__(self, scale):
        super(FrequencyDimension, self).__init__()
        self.scale = scale

    def modified_dimension(self, size, windowsize, stepsize=None):
        raise NotImplementedError()

    def metaslice(self, index, size):
        return FrequencyDimension(self.scale[index])

    def integer_based_slice(self, index):
        if not isinstance(index, frequencyscale.FrequencyBand):
            return index

        return self.scale.get_slice(index)

    def validate(self, size):
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
    """

    def __init__(self, scale, slices):
        super(ExplicitFrequencyDimension, self).__init__()
        if len(scale) != len(slices):
            raise ValueError('scale and slices must have same length')

        self.scale = scale
        self.slices = slices
        self._lookup = dict(zip(self.scale, self.slices))

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
