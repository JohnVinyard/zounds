import numpy as np


def hyperplanes(means, stds, n_planes):
    if len(means) != len(stds):
        raise ValueError('means and stds must have the same length')

    n_features = len(means)
    a = np.random.normal(means, stds, (n_planes, n_features))
    b = np.random.normal(means, stds, (n_planes, n_features))
    plane_vectors = a - b
    return plane_vectors


def simhash(plane_vectors, data):
    output = np.zeros((len(data), len(plane_vectors)), dtype=np.uint8)
    flattened = data.reshape((len(data), -1))
    x = np.dot(plane_vectors, flattened.T).T
    output[np.where(x > 0)] = 1
    return output


def example_wise_unit_norm(x, return_norms=False):
    original_shape = x.shape

    # flatten all dimensions of x, treating the first axis as examples and all
    # other axes as features
    x = x.reshape((len(x), -1))
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    normed = np.divide(x, norms, where=norms != 0)
    normed = normed.reshape(original_shape)

    if return_norms:
        return normed, norms
    else:
        return normed
