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
