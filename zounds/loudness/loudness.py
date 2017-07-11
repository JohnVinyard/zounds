import numpy as np


def log_modulus(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def inverse_log_modulus(x):
    return (np.exp(np.abs(x)) - 1) * np.sign(x)


def decibel(x):
    return 20 * np.log10(x)
