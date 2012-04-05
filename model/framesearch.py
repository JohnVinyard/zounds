from __future__ import division
from abc import ABCMeta,abstractmethod
import math

import numpy as np
from scipy.sparse import lil_matrix

from model import Model
from pattern import DataPattern
from util import pad

from matplotlib import pyplot as plt

class MetaFrameSearch(ABCMeta):
    
    def __init__(self,name,bases,attrs):
        super(MetaFrameSearch,self).__init__(self)
    
    def __getitem__(self,key):
        return self.controller()[key]
    
    def __delitem__(self,key):
        del self.controller()[key]

# TODO: I'm not sure the model module package is the appropriate place for this,
# and/or if it should be Model-derived
class FrameSearch(Model):
    '''
    FrameSearch-derived classes use the frames backing store to provide results
    to queries in the form of sound using one or more stored features.
    '''
    __metaclass__ = MetaFrameSearch
    
    def __init__(self,_id,*features):
        Model.__init__(self)
        self._id = _id
        self.features = features
    
    @classmethod
    def __getitem__(cls,key):
        return cls.controller()[key]
    
    @abstractmethod
    def _build_index(self):
        '''
        Build any data structures needed by this search and persist them 
        somehow
        '''
        pass
    
    def build_index(self):
        self._build_index()
        self.controller().store(self)
    
    @abstractmethod
    def _search(self,frames):
        '''
        Do work
        '''
        pass
    
   
    # KLUDGE: This is a total mess!  Doing on the fly audio extraction should
    # be much easier and nicer than this
    def search(self,query, nresults = 10):
        env = self.env()
        
        if isinstance(query,env.framemodel):
            # The query is a frames instance, so it can be passed to _search
            # directly; the features are already computed
            return self._search(query, nresults = nresults)
        
        if isinstance(query,env.address_class):
            # The query is an address. Use it to get a frames instance which
            # can be passed to _search
            return self._search(env.framemodel[query], nresults = nresults)
        
        # The query wasn't a frames instance, or an address, so we'll assume
        # that it's a numpy array representing raw audio samples
        p = DataPattern(env.newid(),'search','search',query)
        
        fm = env.framemodel
        # build an extractor chain which will compute only the features
        # necessary 
        ec = fm.extractor_chain(p)
        # extract the features into a dictionary
        d = ec.collect()
        
        # turn the dictionary into a numpy recarray
        dtype = []
        for e in ec:
            if 'audio' == e.key or\
             (fm.features.has_key(e.key) and fm.features[e.key].store):
                dtype.append((e.key,e.dtype,e.dim(env)))
        
        l = len(d['audio'])
        r = np.recarray(l,dtype=dtype)
        
        for k,v in d.iteritems():
            if 'audio' == k or\
             (fm.features.has_key(k) and fm.features[k].store):
                r[k] = pad(np.array(v).repeat(ec[k].step, axis = 0),l)
            
        
        # get a frames instance
        frames = fm(data = r)
        return self._search(frames, nresults = nresults)

'''
Minhash search algorithm:

For a minhash algorithm with N hash functions

For a single block, compute the minhash to get N values, V[]

Foreach minhash value, visit the bucket that corresponds to the minhash
function and value, which we'll denote as (N,V) and increment a block's
score by one each time it is encountered.  Once this is complete, the 
M blocks with the highest values are returned. The data structure looks like
[
    // hash function 0
    {
        value : [blocks.....]
    }
    // has function 1
    {
        value : [blocks.....]
    }
    ...
]
'''
class MinHashSearch(FrameSearch):
    
    def __init__(self,_id,feature,step,size):
        FrameSearch.__init__(self,_id,feature)
        self._index = None
        self.step = step
        self.size = size
    
    @property
    def feature(self):
        return self.features[0]

    @property
    def address(self):
        return self._index[0]

    @property
    def index(self):
        return self._index[1]
    
    def _build_index(self):
        env = self.env()
        fc = env.framecontroller
        addresses = []
        index = [{} for s in xrange(self.size)]
        for i,f in enumerate(fc.iter_all(step = self.step)):
            address,frames = f
            addresses.append(address)
            hsh = frames[self.feature]
            hsh = hsh if 1 == len(hsh.shape) else hsh[0]
            print hsh
            for q,h in enumerate(hsh):
                try:
                    index[q][h].append(i)
                except KeyError:
                    index[q][h] = [i]
                    
        self._index = [np.array(addresses),index]
    
    def _search_block(self,hashvalue,nresults):
        d = {}
        index = self.index
        for i,h in enumerate(hashvalue):
            addrs = index[i][h]
            for a in addrs:
                try:
                    d[a] += 1
                except KeyError:
                    d[a] = 1
        items = d.items()
        items.sort(key = lambda i : i[1], reverse = True)
        a = [i[0] for i in items[:nresults]]
        print a
        return self.address[a]
        
    def _search(self,frames, nresults):
        feature = frames[self.feature]
        print '========================================'
        for block in feature[::self.step]:
            addrs = self._search_block(block, 5)
            yield addrs[0]
        
        
        
        
            
            
        