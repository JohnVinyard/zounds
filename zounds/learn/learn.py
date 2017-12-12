from scipy.cluster.vq import kmeans
from featureflow import Node
from preprocess import PreprocessResult, Preprocessor


class KMeans(Preprocessor):
    """

    """
    def __init__(self, centroids=None, needs=None):
        super(KMeans, self).__init__(needs=needs)
        self._centroids = centroids

    def _forward_func(self):
        def x(d, codebook=None):
            import numpy as np
            from scipy.spatial.distance import cdist
            from zounds.core import ArrayWithUnits, IdentityDimension
            l = d.shape[0]
            dist = cdist(d, codebook)
            best = np.argmin(dist, axis=1)
            feature = np.zeros((l, len(codebook)), dtype=np.uint8)
            feature[np.arange(l), best] = 1
            try:
                return ArrayWithUnits(
                    feature, [d.dimensions[0], IdentityDimension()])
            except AttributeError:
                return feature

        return x

    def _backward_func(self):
        def x(d, codebook=None):
            import numpy as np
            indices = np.where(d == 1)
            return codebook[indices[1]]

        return x

    def _process(self, data):
        data = self._extract_data(data)
        codebook, _ = kmeans(data, self._centroids)
        op = self.transform(codebook=codebook)
        inv_data = self.inversion_data(codebook=codebook)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
            data, op, inversion_data=inv_data, inverse=inv, name='KMeans')


class Learned(Node):
    """

    """
    def __init__(self, learned=None, version=None, wrapper=None, needs=None):
        super(Learned, self).__init__(needs=needs)
        self._wrapper = wrapper
        self._learned = learned
        self._version = version

    @property
    def version(self):
        return self._version or self._learned.pipeline.version

    def _process(self, data):
        transformed = self._learned.pipeline\
            .transform(data, wrapper=self._wrapper).data
        yield transformed
