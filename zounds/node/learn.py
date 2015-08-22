from flow import Node
from preprocess import Op, PreprocessResult, Preprocessor
from timeseries import ConstantRateTimeSeries
from scipy.cluster.vq import kmeans
from zounds.learn.nnet.rbm import Rbm, LinearRbm as LinRbm

class KMeans(Preprocessor):
    
    def __init__(self, centroids = None, needs = None):
        super(KMeans, self).__init__(needs = needs)
        self._centroids = centroids
    
    def _process(self, data):
        data = self._extract_data(data)
        codebook, _ = kmeans(data, self._centroids)
        
        def x(d, codebook = None):
            import numpy as np
            from scipy.spatial.distance import cdist
            l = d.shape[0]
            dist = cdist(d,codebook)
            best = np.argmin(dist,axis = 1)
            feature = np.zeros((l,len(codebook)),dtype = np.uint8)
            feature[np.arange(l),best] = 1
            return feature
        
        op = Op(x, codebook = codebook)
        data = op(data)
        yield PreprocessResult(data, op)

class BaseRbm(Preprocessor):
    
    def __init__(\
         self, 
         cls = None, 
         hdim = None, 
         sparsity_target = 0.01, 
         epochs = 100, 
         needs = None):
        
        super(BaseRbm, self).__init__(needs = needs)
        self._hdim = hdim
        self._sparsity = sparsity_target
        self._epochs = epochs
        self._cls = cls
    
    def _process(self, data):
        data = self._extract_data(data)
        rbm = self._cls(\
            indim = data.shape[1], 
            hdim = self._hdim, 
            sparsity_target = self._sparsity)
        rbm.train(data, lambda epoch, error : epoch > self._epochs)
    
        def x(d, rbm = None):
            return rbm(d)
        
        op = Op(x, rbm = rbm)
        data = op(data)
        yield PreprocessResult(data, op)
    
class BinaryRbm(BaseRbm):
    
    def __init__(\
         self, sparsity_target = 0.01, 
         hdim = None, 
         epochs = 100, 
         needs = None):
        
        super(BinaryRbm, self).__init__(\
            cls = Rbm, 
            hdim = hdim, 
            sparsity_target = sparsity_target,
            epochs = epochs, 
            needs = needs)

class LinearRbm(BaseRbm):
    
    def __init__(\
         self, 
         hdim = None,
         sparsity_target = 0.01, 
         epochs = 100, 
         needs = None):
        
        super(LinearRbm, self).__init__(\
            cls = LinRbm, 
            hdim = hdim,
            sparsity_target = sparsity_target, 
            epochs = epochs, 
            needs = needs)

class Learned(Node):
    
    def __init__(self, learned = None, needs = None):
        super(Learned, self).__init__(needs = needs)
        self._learned = learned
    
    def _process(self, data):
        transformed = self._learned(data)
        if isinstance(data, ConstantRateTimeSeries):
            yield ConstantRateTimeSeries(\
                 transformed, 
                 frequency = data.frequency, 
                 duration = data.duration)
        else:
            yield transformed