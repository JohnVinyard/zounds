import frequencyscale
from zounds.core import Dimension


class FrequencyDimension(Dimension):
    def __init__(self, scale):
        super(FrequencyDimension, self).__init__()
        self.scale = scale

    def modified_dimension(self, size, windowsize, stepsize=None):
        raise NotImplementedError()

    def metaslice(self, index, size):
        print 'metaslice', index, size
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