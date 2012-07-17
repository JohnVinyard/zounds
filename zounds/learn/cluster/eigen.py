from __future__ import division
from zounds.learn.learn import Learn
import numpy as np
from zounds.nputil import safe_unit_norm as sun
from scipy.spatial.distance import cdist


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
        dist = cdist(np.array([data]),self.codebook)[0]
        dist[dist == 0] = 1e-3
        return 1 / dist