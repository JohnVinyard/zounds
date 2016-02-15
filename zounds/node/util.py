import os
import numpy as np


def iter_files(base_path):
    for fn in os.listdir(base_path):
        yield os.path.join(base_path, fn)


def process_dir(base_path, process_func):
    for fp in iter_files(base_path):
        process_func(fp)


def sigmoid(a):
    return 1. / (1 + np.exp(-a))


def stochastic_binary(a):
    return a > np.random.random_sample(a.shape)
