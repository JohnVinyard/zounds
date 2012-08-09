from __future__ import division
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from zounds.learn.learn import Learn

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
        l = self.in_data.shape[0]
        dist = cdist(self.in_data,self.codebook)
        best = np.argmin(dist,axis = 1)
        feature = np.zeros((l,len(self.codebook)),dtype = np.uint8)
        feature[best] = 1
        return feature

from zounds.util import flatten2d

# BUG: The problem with this method is that the exemplars are taken from the
# database, so there will be very large activations for these.  Ideally, we'd
# like to resynthesize from the fft means 
class ConvolutionalKMeans(KMeans):
    
    def __init__(self,n_centroids,patch_shape,fft_codebook = False):
        '''
        n_centroids - the number of kmeans clusters to learn
        patch_shape - the 2d shape of sample patches
        fft_codebook - If False (default), the codebook is inferred by finding
                       the input sample with the best matching fft coefficients
                       for each code.  If True, the codebook *is* the fft coefficients.
        '''
        KMeans.__init__(self,n_centroids)
        self._patch_shape = patch_shape if isinstance(patch_shape,tuple) \
                            else (patch_shape,)
        self.fft_codebook = fft_codebook
    
    
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
        if self.fft_codebook:
            self.codebook = codebook
        else:
            # find exemplars in the input data whose coefficients most closely match
            # the codebook fft coefficients, ignoring phase
            dist = cdist(codebook,flatten2d(abs(f)))
            self.codebook = flatten2d(data[np.argmin(dist,1)])
        
    
    
class SoftKMeans(KMeans):
    def __init__(self,n_centroids):
        KMeans.__init__(self,n_centroids)
    
    def __call__(self,data):
        dist = cdist(self.in_data,self.codebook)
        dist[dist == 0] = -1e12
        return 1 / dist
    


class TopNKMeans(KMeans):
    
    def __init__(self,n_centroids,topn):
        KMeans.__init__(self,n_centroids)
        self.topn = topn
    
    def __call__(self,data):
        # For a more in-depth explanation of what's going on here, check out:
        # http://stackoverflow.com/questions/6155649/sort-a-numpy-array-by-another-array-along-a-particular-axis
        dist = cdist(self.in_data,self.codebook)
        # for each example, get the sorted distances from each cluster
        srt = np.argsort(dist,axis = -1)
        # for each example, assign the clsuters with the n lowest distances a 1, 
        # and the rest of the clusters a 0. 
        o = np.ogrid[slice(dist.shape[0]),slice(dist.shape[1])]
        dist[o[0],srt[:,:self.topn]] = 1
        dist[o[0],srt[:,self.topn:]] = 0
        return dist