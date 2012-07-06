from __future__ import division
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from learn.learn import Learn
from sklearn.decomposition import PCA as SKPCA

# KLUDGE: I've added indim and hdim so this class can be used 
        # as a NeuralNetwork-derived class
class KMeans(Learn):
    
    def __init__(self,n_centroids):
        Learn.__init__(self)
        self.n_centroids = n_centroids
        self.codebook = None
        
    
    @property
    def indim(self):
        return self.codebook.shape[1]
    
    @property
    def hdim(self):
        return self.codebook.shape[0]
    
    @property
    def dim(self):
        return self.hdim
    
    def train(self,data,stopping_condition):
        self._indim = data.shape[1]
        codebook,distortion = kmeans(data,self.n_centroids)
        self._hdim = len(codebook)
        self.codebook = codebook
    
    def __call__(self,data):
        dist = cdist(np.array([data]),self.codebook)[0]
        best = np.argmin(dist)
        feature = np.zeros(len(self.codebook),dtype = np.uint8)
        feature[best] = 1
        return feature
    
class SoftKMeans(KMeans):
    def __init__(self,n_centroids):
        KMeans.__init__(self,n_centroids)
    
    def __call__(self,data):
        dist = cdist(np.array([data]),self.codebook)[0]
        dist[dist == 0] = -1e12
        return 1 / dist
    

class ThresholdKMeans(SoftKMeans):
    def __init__(self,n_centroids,threshold):
        SoftKMeans.__init__(self,n_centroids)
        self._threshold = threshold

    def __call__(self,data):
        if data.sum() < self._threshold:
            return np.zeros(len(self.codebook))

        dist = cdist(np.array([data]),self.codebook)[0]
        dist[dist == 0] = -1e12
        return 1 / dist

        
# KLUDGE: This doesn't belong here
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