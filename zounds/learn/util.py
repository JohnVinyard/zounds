import numpy as np

def sigmoid(a):
    return 1. / (1 + np.exp(-a))


def stochastic_binary(a):
    return a > np.random.random_sample(a.shape)