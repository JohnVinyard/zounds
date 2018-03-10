from __future__ import division
import numpy as np
from featureflow import Node


def log_modulus(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def inverse_log_modulus(x):
    return (np.exp(np.abs(x)) - 1) * np.sign(x)


def decibel(x):
    return 20 * np.log10(x)


def mu_law(x, mu=255):
    s = np.sign(x)
    x = np.abs(x)
    return s * (np.log(1 + (mu * x)) / np.log(1 + mu))


def inverse_mu_law(x, mu=255):
    s = np.sign(x)
    x = np.abs(x)
    x *= np.log(1 + mu)
    x = (np.exp(x) - 1) / mu
    return x * s


def inverse_one_hot(x, axis=-1):
    n_categories = x.shape[axis]
    indices = np.argmax(x, axis=axis).astype(np.float32)
    indices /= float(n_categories)
    indices = (indices - 0.5) * 2
    return indices


def instance_scale(x, axis=-1, epsilon=1e-8):
    return x / (x.max(axis=axis) + epsilon)


def unit_scale(x, axis=None):
    scaled = x - x.min(axis=axis, keepdims=True)
    mx = scaled.max(axis=axis, keepdims=True)
    scaled = np.divide(scaled, mx, where=max != 0)
    return scaled


class MuLaw(Node):
    def __init__(self, mu=255, needs=None):
        super(MuLaw, self).__init__(needs=needs)
        self.mu = mu

    def _process(self, data):
        yield mu_law(data, mu=self.mu)


class LogModulus(Node):
    def __init__(self, factor=1, needs=None):
        super(LogModulus, self).__init__(needs=needs)
        self.factor = factor

    def _process(self, data):
        yield log_modulus(data * self.factor)
