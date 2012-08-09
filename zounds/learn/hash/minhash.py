import numpy as np

from zounds.learn.learn import Learn


class MinHash(Learn):
    '''
    A hash that approximates the Jaccard distance between two binary feature 
    vectors.
    
    http://en.wikipedia.org/wiki/MinHash
    '''
    def __init__(self,size,nhashes):
        Learn.__init__(self)
        self.size = size
        self.nhashes = nhashes
        self.hashes = None
    
    def train(self,data,stopping_condition):
        '''
        Pick n hash functions, i.e., permutations of the vocabulary
        '''
        self.hashes = \
            [np.random.permutation(self.size) for h in range(self.nhashes)]
    
    def __call__(self,data):
        '''
        Produce a hash by finding the first non-zero "word" from each
        permutation of the input
        '''
        a = np.ndarray((data.shape[0],self.nhashes))
        # assign a special value that means, "there were no nonzero values"
        a[...] = self.size + 1
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