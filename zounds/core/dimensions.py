class Dimension(object):
    """
    Common base class representing one dimension of a numpy array.  Sub-classes
    can define behavior making custom slices (e.g., time spans or
    frequency bands) possible.

    Implementors are primarily responsible for determining how custom slices
    are transformed into integer indexes and slices that numpy can use directly.

    See Also:
        :class:`IdentityDimension`
        :class:`~zounds.timeseries.TimeDimension`
        :class:`~zounds.spectral.FrequencyDimension`
    """

    def __init__(self):
        super(Dimension, self).__init__()

    def modified_dimension(self, size, windowsize, stepsize=None):
        raise NotImplementedError()

    def metaslice(self, index, size):
        """
        Produce a new instance of this dimension, given a custom slice
        """
        return self

    def integer_based_slice(self, index):
        """
        Subclasses define behavior that transforms a custom, user-defined slice
        into integer indices that numpy can understand

        Args:
            index (custom slice): A user-defined slice instance
        """
        raise NotImplementedError()

    def validate(self, size):
        """
        Subclasses check to ensure that the dimensions size does not validate
        any assumptions made by this instance
        """
        pass


class IdentityDimension(Dimension):
    """
    A custom dimension that does not transform indices in any way, simply acting
    as a pass-through.

    Examples:
        >>> from zounds import ArrayWithUnits, IdentityDimension
        >>> import numpy as np
        >>> data = np.zeros(100)
        >>> arr = ArrayWithUnits(data, [IdentityDimension()])
        >>> sliced = arr[4:6]
        >>> sliced.shape
        (2,)
    """

    def __init__(self):
        super(IdentityDimension, self).__init__()

    def modified_dimension(self, size, windowsize, stepsize=None):
        if windowsize == slice(None) or (size / windowsize) == 1:
            yield IdentityDimension()
        else:
            raise ValueError()

    def integer_based_slice(self, index):
        return index

    def __eq__(self, other):
        return self.__class__ == other.__class__
