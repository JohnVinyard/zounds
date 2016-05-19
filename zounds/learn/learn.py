from scipy.cluster.vq import kmeans

from featureflow import Node
from preprocess import PreprocessResult, Preprocessor
from rbm import Rbm, RealValuedRbm
from zounds.timeseries import ConstantRateTimeSeries


class KMeans(Preprocessor):
    def __init__(self, centroids=None, needs=None):
        super(KMeans, self).__init__(needs=needs)
        self._centroids = centroids

    def _forward_func(self):
        def x(d, codebook=None):
            import numpy as np
            from scipy.spatial.distance import cdist
            l = d.shape[0]
            dist = cdist(d, codebook)
            best = np.argmin(dist, axis=1)
            feature = np.zeros((l, len(codebook)), dtype=np.uint8)
            feature[np.arange(l), best] = 1
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


class BaseRbm(Preprocessor):
    def __init__(
            self,
            cls=None,
            hdim=None,
            sparsity_target=0.01,
            epochs=100,
            learning_rate=0.1,
            needs=None):
        super(BaseRbm, self).__init__(needs=needs)
        self._hdim = hdim
        self._sparsity = sparsity_target
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._cls = cls

    def _forward_func(self):
        def x(d, rbm=None):
            return rbm(d)

        return x

    def _backward_func(self):
        def x(d, rbm=None):
            return rbm.fromfeatures(d, binarize=False)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        rbm = self._cls(
                indim=data.shape[1],
                hdim=self._hdim,
                sparsity_target=self._sparsity,
                learning_rate=self._learning_rate)
        rbm.train(data, lambda epoch, error: epoch > self._epochs)
        op = self.transform(rbm=rbm)
        inv_data = self.inversion_data(rbm=rbm)
        inv = self.inverse_transform()
        data = op(data)
        yield PreprocessResult(
                data,
                op,
                inversion_data=inv_data,
                inverse=inv,
                name=self.__class__.__name__)


class BinaryRbm(BaseRbm):
    def __init__(
            self, sparsity_target=0.01,
            hdim=None,
            epochs=100,
            learning_rate=0.1,
            needs=None):
        super(BinaryRbm, self).__init__(
                cls=Rbm,
                hdim=hdim,
                sparsity_target=sparsity_target,
                epochs=epochs,
                learning_rate=learning_rate,
                needs=needs)


class LinearRbm(BaseRbm):
    def __init__(
            self,
            hdim=None,
            sparsity_target=0.01,
            epochs=100,
            learning_rate=0.001,
            needs=None):
        super(LinearRbm, self).__init__(
                cls=RealValuedRbm,
                hdim=hdim,
                sparsity_target=sparsity_target,
                epochs=epochs,
                learning_rate=learning_rate,
                needs=needs)


class Learned(Node):
    def __init__(self, learned=None, needs=None):
        super(Learned, self).__init__(needs=needs)
        self._learned = learned

    @property
    def version(self):
        return self._learned.pipeline.version

    def _process(self, data):
        transformed = self._learned.pipeline.transform(data).data
        if isinstance(data, ConstantRateTimeSeries):
            yield ConstantRateTimeSeries(
                    transformed,
                    frequency=data.frequency,
                    duration=data.duration)
        else:
            yield transformed
