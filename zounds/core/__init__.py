"""
The core module introduces the key building blocks of the representations zounds
deals in: :class:`ArrayWithUnits`, a :class:`numpy.ndarray`-derived class that
supports semantically meaningful indexing, and :class:`Dimension`, a common
base class for custom, user-defined dimensions.
"""

from dimensions import Dimension, IdentityDimension
from axis import ArrayWithUnits

__all__ = [Dimension, IdentityDimension, ArrayWithUnits]
