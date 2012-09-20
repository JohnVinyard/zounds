import numpy as np

from zounds.learn.learn import Learn
from zounds.util import tostring


class MinHash(Learn):
    '''
    The `min-hash <http://en.wikipedia.org/wiki/MinHash>`_ algorithm approximates
    the Jaccard distance between two binary feature vectors.
    
    A single hash function is simply a random permutation of all integers from
    0 to :code:`len(vector) - 1`, i.e., indices of individual features.  Given
    a feature vector, the hash function re-orders the vector according to its
    permutation, and returns the lowest index with an "on" bit.
    
    Many such hashes are created, and the output of each is concatenated into a
    vector of integers.
    
    When comparing the output of the min-hash algorithm to approximate the 
    similarity of two binary feature vectors, it's important to remember that
    hamming distance is the only metric that makes sense, since the integers
    themselves carry no special meaning.
    
    Note that the "training" phase for the min-hash algorithm merely consists of
    picking n random permutations.  These are persisted for the life of an instance,
    so hash values are always reproducible.
    '''
    
    def __init__(self,size,nhashes):
        '''__init__
        
        :param size: the dimension of input vectors
        
        :param nhashes: The number of hash functions, i.e., permutations of the \
        indices of input vectors, to create.
        '''
        
        Learn.__init__(self)
        self.size = size
        self.nhashes = nhashes
        self.hashes = None
    
    def train(self,data,stopping_condition):
        '''train
        
        Pick n hash functions, i.e., permutations of the vocabulary
        
        :param data: data is ignored.
        
        :param stopping_condition: stopping condition is ignored.
        '''
        self.hashes = \
            [np.random.permutation(self.size) for h in range(self.nhashes)]
    
    def __call__(self,data):
        '''__call__
        
        Produce a hash by finding the first non-zero "word" from each
        permutation of the input
        
        :param: a two-dimensional numpy array containing binary feature vectors \
        to hash
        
        '''
        a = np.ndarray((data.shape[0],self.nhashes))
        # assign a special value that means, "there were no nonzero values"
        a[...] = self.size + 1
        
        # TODO: I bet this for loop can be gotten rid of...
        for i,h in enumerate(self.hashes):
            # get the nonzero indices for this permutation
            nz = np.nonzero(data[h])
            # get the unique values from the first dimension of nz, and the
            # indices at which they appear. This is necessary because we 
            # want to know the *first* nonzero index, and not all of them.
            unique,indices = np.unique(nz[0],return_index = True)
            # For every example, for this permutation, assign the index of the
            # first non-zero value, if there was one
            a[:,i][unique] = nz[1][indices]
        return a
    
    def __repr__(self):
        return tostring(self,feature_dim = self.size,n_hashes = self.nhashes)
    
    def __str__(self):
        return self.__repr__()