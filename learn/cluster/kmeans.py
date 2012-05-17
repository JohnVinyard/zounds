import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from learn.learn import Learn

class KMeans(Learn):
    
    def __init__(self,n_centroids):
        Learn.__init__(self)
        self.n_centroids = n_centroids
        self.codebook = None
    
    def train(self,data,stopping_condition):
        codebook,distortion = kmeans(data)
        self.codebook = codebook
    
    def __call__(self,data):
        dist = cdist(np.array([data]),self.codebook)[0]
        best = np.argmax(dist)
        feature = np.zeros(len(self.codebook),dtype = np.uint8)
        feature[best] = 1
        return feature