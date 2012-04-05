import numpy as np

from learn.learn import Learn


class MinHash(Learn):
    '''
    A hash that approximates the Jaccard distance.
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
        a = np.ndarray(self.nhashes)
        for i,h in enumerate(self.hashes):
            try:
                a[i] = np.where(data[h] == 1)[0][0]
            except IndexError:
                a[i] = self.size + 1
        return a
            