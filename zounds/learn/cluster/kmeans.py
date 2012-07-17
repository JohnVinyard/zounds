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

from util import flatten2d

# BUG: The problem with this method is that the exemplars are taken from the
# database, so there will be very large activations for these.  Ideally, we'd
# like to resynthesize from the fft means 
class ConvolutionalKMeans(KMeans):
    
    def __init__(self,n_centroids,patch_shape):
        KMeans.__init__(self,n_centroids)
        self._patch_shape = patch_shape if isinstance(patch_shape,tuple) \
                            else (patch_shape,)
    
    def train(self,data,stopping_condition):
        '''
        The idea here is to avoid redundant codes, i.e., codes that are simply
        translated.  The plan to avoid this is as follows:
        
        1) take an fft of input patches
        2) perform kmeans clustering on the real-valued (phase is discarded) fft coefficients
        3) find the best exemplars of the resulting clusters in the fft data from step one.
        4) the corresponding patches from the input data are our codebook
        '''
        data = data.reshape((len(data),) + self._patch_shape)
        # we always want to treat the first dimension as examples, and compute
        # an n-dimensional fft over the remaining dimensions
        axes = -np.arange(len(data.shape))[1:][::-1]
        # take an n-dimensional fft of each data example
        f = np.fft.rfftn(data,axes = axes)
        # compute k-means on the real-valued (discarded phase) fft coefficients
        codebook,distortion = kmeans(flatten2d(abs(f)),self.n_centroids)
        self._hdim = len(codebook)
        # find exemplars in the input data whose coefficients most closely match
        # the codebook fft coefficients, ignoring phase
        dist = cdist(codebook,flatten2d(abs(f)))
        self.codebook = flatten2d(data[np.argmin(dist,1)])
        
    
    
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