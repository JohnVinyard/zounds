from __future__ import division

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as SKPCA

from zounds.learn.learn import Learn
from zounds.nputil import safe_unit_norm as sun

class Eigenvectors(Learn):
    
    def __init__(self,ncodes):
        Learn.__init__(self)
        self._ncodes = ncodes
        self.codebook = None
    
    def train(self,data,stopping_condition):
        cov = np.dot(data.T,data)
        values,vectors = np.linalg.eig(cov)
        self.codebook = vectors.T[:self._ncodes]
    
    def __call__(self,data):
        dist = cdist(self.in_data,self.codebook)
        dist[dist == 0] = 1e-3
        return 1 / dist


class PCA(Learn):
    
    def __init__(self,n_dim):
        Learn.__init__(self)
        self.n_dim = n_dim
        self._pca = None
    
    def train(self,data,stopping_condition):
        self._pca = SKPCA(self.n_dim)
        self._pca.fit(data)
    
    def __call__(self,data):
        return self._pca.transform(data)