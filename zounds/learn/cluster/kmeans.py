from __future__ import division

import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist

from zounds.learn.learn import Learn
from zounds.nputil import flatten2d
from zounds.util import tostring
from zounds.visualize import plot_series

# KLUDGE: I've added indim and hdim so this class can be used 
# as a NeuralNetwork-derived class
class KMeans(Learn):
    '''
    Perform `k-means clustering <http://en.wikipedia.org/wiki/K-means_clustering>`_
    on data. 
    
    K-means attempts to partition data into n clusters by minimizing the 
    distance of each data example from the mean of its assigned cluster, for 
    all clusters.
    
    This class is really just a thin wrapper around 
    `scipy.cluster.vq.kmeans <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans>`_
    '''
    
    def __init__(self,n_centroids,guess = None):
        '''__init__
        
        :param n_centroids:  The desired number of clusters. Note that a slightly \
        lower number of clusters may be returned, in some cases.
        
        :param guess: Initial guess for the cluster centers.  It should be a two-\
        dimensional array whose first dimension is equal to :code:`n_centroids` \
        and whose second dimension is equal to the dimension of input features \
        vectors.
        '''
        Learn.__init__(self)
        self.n_centroids = n_centroids
        self.codebook = None
        self.guess = guess
    
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
        '''train
        
        :param data: The examples to cluster
        
        :param stopping_condition:  A callable which takes no arguments
        '''
        self._indim = data.shape[1]
        centroids = self.guess if None is not self.guess else self.n_centroids
        codebook,distortion = kmeans(data,centroids)
        self._hdim = len(codebook)
        self.codebook = codebook
    
    def __call__(self,data):
        '''__call__
        
        Transform the data by mapping each input example to the nearest cluster
        center
        
        :param data: A two-dimensional numpy array of examples to be mapped to \
        cluster centers.
        
        :returns: A numpy arrray whose shape is :code:`(len(data),n_centroids)`.  \
        Each row consists of all zeros, except for a one in the position of the \
        best matching cluster center.
        '''
        l = data.shape[0]
        dist = cdist(data,self.codebook)
        best = np.argmin(dist,axis = 1)
        feature = np.zeros((l,len(self.codebook)),dtype = np.uint8)
        feature[np.arange(l),best] = 1
        return feature
    
    def __repr__(self):
        return tostring(self,n_centroids = self.n_centroids)
    
    def __str__(self):
        return self.__repr__()
    
    def view_codes(self,path):
        import os
        plot_series(self.codebook,os.path.join(path,'codes'))


# TODO: Add a sparsify option which will zero out values below the row's mean,
# after the inverse has been taken.
class SoftKMeans(KMeans):
    '''
    :py:class:`SoftKMeans`' training phase is identical to :py:class:`KMeans`, but
    its :py:meth:`__call__` method returns the inverse of the distance to *every*
    cluster center, instead of a "one-hot" encoding.  This should give a richer, 
    more nuanced description of input examples.
    '''
    
    def __init__(self,n_centroids):
        KMeans.__init__(self,n_centroids)
    
    def __call__(self,data):
        '''__call__
        
        Transform the data by returning the inverse of the distance to *every*
        cluster center.
        
        :param data: A two-dimensional numpy array of examples to be mapped to \
        cluster centers.
        
        :returns: A numpy array whose shape is :code:`(len(data),n_centroids)`.  \
        Each row is :code:`1 / distance` from that example to every cluster center.
        '''
        dist = cdist(data,self.codebook)
        dist[dist == 0] = -1e12
        return 1 / dist


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
        
    
class TopNKMeans(KMeans):
    
    def __init__(self,n_centroids,topn):
        KMeans.__init__(self,n_centroids)
        self.topn = topn
    
    def __call__(self,data):
        # For a more in-depth explanation of what's going on here, check out:
        # http://stackoverflow.com/questions/6155649/sort-a-numpy-array-by-another-array-along-a-particular-axis
        dist = cdist(data,self.codebook)
        # for each example, get the sorted distances from each cluster
        srt = np.argsort(dist,axis = -1)
        # for each example, assign the clsuters with the n lowest distances a 1, 
        # and the rest of the clusters a 0. 
        o = np.ogrid[slice(dist.shape[0]),slice(dist.shape[1])]
        dist[o[0],srt[:,:self.topn]] = 1
        dist[o[0],srt[:,self.topn:]] = 0
        return dist

class KMeansPP(SoftKMeans):
    '''
    A simpler KMeans++ variant that simply maximizes the distance from all
    existing centroids when choosing the next centroid, instead of choosing
    randomly after assigning weighted probabilities.
    '''
    def __init__(self,n_centroids):
        KMeans.__init__(self,n_centroids)
    
    def train(self,data,stopping_condition):
        self._indim = data.shape[1]
        dl = data.shape[0]
        codebook = np.zeros((self.n_centroids,self._indim))
        seed = np.random.randint(dl)
        codebook[0] = data[seed]
        for i in range(1,self.n_centroids):
            dist = cdist(codebook[:i],data)
            # pick the example that maximizes the distance from *all* codes
            # picked thus far
            nc = np.argmax(dist.sum(0))
            print nc
            codebook[i] = data[nc]
        
        self._hdim = len(codebook)
        self.codebook = codebook
            
    