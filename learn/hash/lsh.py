import struct

import numpy as np
from scipy.cluster.vq import whiten,kmeans
from scipy.spatial.distance import cdist
from bitarray import bitarray

from learn.learn import Learn

class Lsh(Learn):
    
    def __init__(self,size,nhashes = 32):
        Learn.__init__(self)
        self.size = size
        self.nhashes = nhashes
        self.hashes = None
    
    def train(self,data,stopping_condition):
        '''
        Assume centered data that is in a completely randomized order.
        Look for hash functions (example data) that approximately splits
        randomly selected data in half.
        '''
        
        # This worked well for bark bands
        #data = whiten(data)
        #codebook,distortion = kmeans(data,self.nhashes)
        #self.hashes = codebook
        
        
        '''
        hashes = []
        l = len(data) - 1
        for d in data:
            nsamples = 100 if l < 100000 else 1000
            ei = np.random.random_integers(0,l,nsamples)
            h = np.sign(np.dot(data[ei],d))
            avg = np.mean(h)
            print avg
            if avg > .4 and avg < .6:
                print 'Found suitable exemplar'    
                hashes.append(d)
                
            if self.nhashes == len(hashes):
                self.hashes = np.array(hashes)
                return
            
        raise ValueError('Not enough sufficient exemplars in the input')
        '''
        
        '''
        hashes = [data[0]]
        for d in data[1:]:
            dist = cdist(np.array([d]),np.array(hashes),'hamming')[0]
            if np.all(dist > .05):
                print 'Found exemplar %i' % len(hashes)
                hashes.append(d)
                
            if self.nhashes == len(hashes):
                self.hashes = np.array(hashes)
                return
        
        raise ValueError('Not enough sufficient exemplars in the input')
        '''
        
        '''
        l = len(data)
        i = 0
        nsamples = 1000
        while i < 1000:
            hashes = np.random.random_integers(0,l,self.nhashes)
            samples = np.random.random_integers(0,l,nsamples)
            results = np.zeros((nsamples,self.nhashes))
            for q in range(nsamples):
                results[q] = np.dot(hashes,samples[q])
            results = np.sign(results)
            avg = results.mean()
            print avg
            if avg > .4 and avg < .6:
                self.hashes = hashes
                return 
            i+=1
        
        
        raise ValueError('Could not find suitable hash function')
        '''
        nsamples = 10000
        best_score = 0
        best = None
        l = len(data) - 1
        for i in range(500):
            self.hashes = data[np.random.random_integers(0,l,self.nhashes)]
            samples = data[np.random.random_integers(0,l,nsamples)]
            results = np.zeros(nsamples)
            
            for q in range(nsamples):
                results[q] = self(samples[q])
            
            unique = np.unique(results)
            lu = len(unique)
            print '%i unique values out of %i' % (lu,nsamples) 
            
            if lu > best_score:
                best_score = lu
                best = self.hashes
        
        self.hashes = best  
            
        
        
    
    def __call__(self,data):
        arr = np.sign(np.dot(self.hashes,data))
        b = bitarray()
        b.extend(arr > 0)
        # KLUDGE: The datatype must correspond to nhashes
        r = struct.unpack('L',b.tobytes())[0]
        print r
        return r