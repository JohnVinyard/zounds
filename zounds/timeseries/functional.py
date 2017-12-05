import numpy as np
from zounds.loudness import mu_law, inverse_mu_law
from zounds.core import ArrayWithUnits, IdentityDimension


def categorical(x, mu=255):
    """
    Mu-law compress a block of audio samples, and convert them into a
    categorical distribution
    """
    # normalize the signal
    mx = x.max()
    x = np.divide(x, mx, where=mx != 0)

    # mu law compression
    x = mu_law(x)

    # translate and scale to [0, 1]
    x = (x - x.min()) * 0.5

    # convert to the range [0, 255]
    x = (x * mu).astype(np.uint8)

    # create the array to house the categorical representation
    c = np.zeros((np.product(x.shape), mu + 1), dtype=np.uint8)
    c[np.arange(len(c)), x.flatten()] = 1

    return ArrayWithUnits(
        c.reshape(x.shape + (mu + 1,)),
        x.dimensions + (IdentityDimension(),))


def inverse_categorical(x, mu=255):
    """
    Invert categorical samples
    """
    flat = x.reshape((-1, x.shape[-1]))
    indices = np.argmax(flat, axis=1).astype(np.float32)
    indices = (indices / mu) - 0.5
    return inverse_mu_law(indices, mu=mu).reshape(x.shape[:-1])
