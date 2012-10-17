from __future__ import division

import numpy as np
from scipy.spatial.distance import cdist

from zounds.learn.learn import Learn
from zounds.util import tostring

class Stopping(object):
    '''
    Stopping condition for self-organizing map training
    '''
    
    def __init__(self,epoch):
        '''__init__
        
        :param epoch: The epoch (one full pass over the data) after which \ 
        training is considered complete
        '''
        self._epoch = epoch
    
    def __call__(self,epoch):
        if epoch > self._epoch:
            return True
        
        return False

class Som(Learn):
    '''
    A two-dimensional 
    `self-organizing map <http://en.wikipedia.org/wiki/Self-organizing_map>`_.
    implementation.
    
    A self-organizing map projects high-dimensional vectors topologically onto a
    lower (two-dimensional, in this case) space.  Similar items (in a euclidean
    sense) can be found in the same neighborhood of the map, once it has been
    trained.
    
    This implementation requires that the map be two-dimensional and square.
    '''
    
    def __init__(self,size,fdim,lasso_size = 18, lasso_decay = 0.8, 
                 gravity = 0.1, gravity_decay = 0.9):
    
        '''__init__
        
        :param size: The number of cells on one edge of the map.  This \
        implementation requires that the map be square, so the number of cells \
        will be :code:`size ** 2`.
        
        :param fdim: The dimension of input data vectors.
        
        :param lasso_size:  Used only during training.  The radius of a circle \
        which is centered on the nearest cell to a given  sample.  Every cell \
        in this region will be dragged closer to the sample.
        
        :param lasso_decay:  The percentage by which the "lasso" will shrink on \
        each epoch training.
        
        :param gravity:  Cells within the "lasso" will be pulled this percentage \
        of their distance from the current training example closer to the sample.
        
        :param gravity_decay: The percentage by which the gravity will decrease \
        on each epoch of training.
        
        '''
        
        Learn.__init__(self)
        self._fdim = fdim
        self._size = size
        self._ncells = size * size
        self._lasso_size = lasso_size
        self._lasso_decay = lasso_decay
        self._gravity = gravity
        self._gravity_decay = gravity_decay
        self._weights = np.zeros((size,size,fdim))
    
    def __repr__(self):
        return tostring(self,feature_dim = self._fdim,n_centroids = self._ncells)
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def codebook(self):
        return self._weights.reshape((self._ncells,self._fdim))
    
    def _init_weights(self,samples):
        '''
        Initialize the cell's weights by
        randomly sampling from the input examples.
        This
        '''
        
        # shuffle the indices of samples
        indices = np.random.permutation(np.arange(len(samples)))
        
        # take the first size indices from samples, and reshape
        # this way, the weights are already within the realm of
        # possibility
        self._weights = np.array(samples[indices[:self._ncells]]\
            .reshape((self._size,self._size,self._fdim))\
            .copy(),dtype=np.double)
        
    def _nearest_cell(self,sample):
        '''
        Return the row and column indices
        of the cell nearest to the example.
        '''
        flat = self._weights.reshape(self._ncells,self._fdim).copy()
        s = np.array([sample])
        # dist is the first (and only) column
        dist = cdist(flat,s)[:,0]
        # choose the flattened index of the closest cell
        index = np.argmin(dist)
        # return row and column indices
        return [int(index / self._size),index % self._size]

    def nearest_cell(self,sample):
        return self._nearest_cell(sample)

    def nearest_cell_flat(self,sample):
        r,c = self.nearest_cell(sample)
        return (r*self._size) + c
    
    def _lasso(self,r,c,size):
        '''
        Return a square window from weights,
        approximately centered on r,c.
        Note that the square might "fall off" one
        or both edges of the SOM
        '''
        sp1 = size + 1
        return slice(r - size , r + sp1), slice(c - size , c + sp1)

    def _rlasso(self,r,c,size):
        '''
        Modify a call to _lasso so that negative
        weights are zeroed. Negative indices don't
        seem to work for multidimensional arrays
        '''
        rsl,csl = self._lasso(r,c,size)
        r_start = 0 if rsl.start < 0 else rsl.start
        c_start = 0 if csl.start < 0 else csl.start
        return slice(r_start,rsl.stop),slice(c_start,csl.stop)

    def map(self,samples):
        '''
        Given a sample set, Display the number of
        samples that fall into each cell. This assumes
        that the SOM has been initialized and trained
        '''
        wm = np.zeros((self._size,self._size))
        for s in samples:
            r,c = self._nearest_cell(s)
            wm[r][c] += 1

    def _hamming(self,size,gravity):
        '''
        Return a 2d hamming window
        '''
        # the full diameter of the uncropped
        # 2d hamming window function we'll be 
        # using to lessen the effects of "gravity"
        diameter = (size * 2) + 1
        # the 1d window. we'll transpose it
        # in order to get a 2d representation
        w = np.hamming(diameter)
        # the 2d window
        return np.outer(w,w) * gravity

    def _warp(self,rsl,csl,size,t,g,w):

        # negative start indices don't seem to work with 
        # multidimensional np arrays the way they do
        # with single dimensional ones
        r_start = 0 if rsl.start < 0 else rsl.start
        c_start = 0 if csl.start < 0 else csl.start
        sl = self._weights[r_start : rsl.stop, c_start : csl.stop]
        real_slice = slice(r_start,rsl.stop),slice(c_start,csl.stop)

        diameter = (size * 2) + 1
        # the distance of every cell in the slice
        # from the sample
        dr = t - sl
        
        # the start index for the window's row slice
        wrsls = 0 if rsl.start >= 0 else -rsl.start
        #  how much does the slice overflow row-wise?
        rof = self._size - rsl.stop
        # the stop index for the window's row slice
        wrsle = rof if rof < 0 else diameter

        # the start index for the window's column slice
        wcsls = 0 if csl.start >= 0 else -csl.start
        # how much does the slice overflow column-wise?
        cof = self._size - csl.stop
        # the stop index for the window's column slice
        wcsle = cof if cof < 0 else diameter
        
        dr *= w[wrsls : wrsle, wcsls : wcsle,np.newaxis]
        # return the slice with non-negative indices that 
        # we've computed, along with the warped slice of
        # the weights matrix
        newsl = sl + dr
        return real_slice,newsl
    
    def _update(self,sample,r,c,lasso_size,gravity,filename = None):
        sl = self._lasso(r,c,lasso_size)
        window = self._hamming(lasso_size,gravity)
        real_sl,warped = self._warp(sl[0],sl[1],lasso_size,sample,gravity,window)
        self._weights[real_sl] = warped
    
    
    def train(self,data,stopping_condition):
        '''train
        
        :param data: A two-dimensional numpy array of training examples
        
        :param stopping_condition: a callable which takes the current epoch as \
        its only argument.
        '''
        self._train(data,
                    stopping_condition,
                    self._lasso_size,
                    self._lasso_decay,
                    self._gravity,
                    self._gravity_decay)
    
    def _train(self,
              samples, 
              stopping_condition,
              lasso_size = 18, 
              lasso_decay = 0.8, 
              gravity = 0.1, 
              gravity_decay = 0.9):
        '''
        Train the SOM on samples for iterations.
        
        lasso_size is the width (and height) of a box
        which is centered on the nearest cell to a given 
        sample.  Every cell in this region will be dragged
        closer to the sample.

        lasso_decay is the amount by which the lasso shrinks
        on each full pass over the samples

        gravity is the amount by which each cell in the lasso
        is pulled closer to the sample

        gravity_spatial decay is the amount by which gravity
        decays as it nears the edge of the lasso

        gravity_decay is the amount by which gravity decays
        on each full pass over the samples
        '''
        

        self._init_weights(samples)
        ls = lasso_size
        ld = lasso_decay
        g = gravity
        gd = gravity_decay
        
        i = 0
        while not stopping_condition(i):
            hamming = self._hamming(ls,g)
            for q,s in enumerate(samples):
                r,c = self._nearest_cell(s)
                sl = self._lasso(r,c,ls)
                real_slice,warped = self._warp(sl[0],
                                               sl[1],
                                               ls,
                                               s,
                                               g,
                                               hamming)
                                    
                self._weights[real_slice] = warped
                if not q % 100:
                    print 'iteration %d, sample %d, lasso %d, gravity %1.4f' % (i,q,ls,g)
            # update the parameters, i.e.
            # lessen the lasso size and gravity
            ls *= ld
            ls = int(round(ls))
            if ls < 1:
                ls = 1
            g *= gd
            self._lasso_size = ls
            self._lasso_decay = ld
            self._gravity = g
            self._gravity_decay = gd
            i+= 1
    
    
    
    def __call__(self,data):
        '''__call__
        
        :param data: A two-dimensional numpy array of samples to be mapped to \
        the best matching cell.
        
        :returns: The inverse of the distance from each example to each cell.  \
        Note that cell addresses are flattened.  To obtain the two dimensional \
        address of a cell: :code:`row_number = cell_number // self._size` and \
        :code:`column_number = cell_number % self._size`.
        
        '''
        dist = cdist(data,self.codebook)
        dist[dist == 0] = 1e-3
        return 1 / dist