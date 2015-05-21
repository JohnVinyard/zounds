from flow import Node
from preprocess import Op, PreprocessResult, Preprocessor
from scipy.cluster.vq import kmeans

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