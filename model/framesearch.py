from __future__ import division
from abc import ABCMeta,abstractmethod
import math

import numpy as np
from scipy.sparse import lil_matrix

from model import Model
from pattern import DataPattern
from util import pad

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


class BinaryFeatureSearch(FrameSearch):
    
    def __init__(self,_id,feature,step):
        FrameSearch.__init__(self,_id,feature)
        self._index = None
        self.step = step
    
    @property
    def feature(self):
        return self.features[0]
    
    @property
    def addresses(self):
        return self._index[0]
    
    @property
    def words(self):
        return self._index[1]
    
    def _extract_data(self,f):
        '''
        Extract the "words" present (non-zero items), as well as the density
        of words
        '''
        return np.nonzero(f)[0]
    
    def _build_index(self):
        env = self.env()
        fc = env.framecontroller
        extractor = self.feature.extractor()
        # the dimension of the binary feature vector
        dim = extractor.dim(env)
        # the number of entries in the index
        # KLUDGE: How do I know *exactly* how long to make the sparse matrix
        # rows *before* iterating over the addresses?
        nentries = int((math.ceil(len(fc) / self.step) * 1.1))
        addresses = []
        index = lil_matrix((dim,nentries), dtype = np.int8)
        for i,f in enumerate(fc.iter_all(step = self.step)):
            address,frames = f
            addresses.append(address)
            # extract the words, i.e., the features that are "on"
            words = self._extract_data(frames[self.feature])
            print i,len(words)
            # set the corresponding features in the index to "on"
            index[[words],i] = 1
        
        index = index.tocsr()
        self._index = [np.array(addresses),index]
            
    
    
    def _top_candidates(self,words,topn = 40):
        index = self.words
        # Given the words, get a score for every block
        # in the db, i.e., Determine how many words
        # each block has in common with the example
        s = index[words,:].sum(0)
        a = np.asarray(s)[0]
    
        # Sort blocks according to score, keeping
        # the top n scores
        best = np.argsort(a)[-topn*10:]
        
        # For the n best blocks, find out how many
        # total words each has
        c = index[:,best].sum(0)
        c = np.asarray(c)[0]
    
        # Get the difference in word count between
        # the example and the n best blocks
        diff = np.abs(c - len(words))
        bestcount = np.argsort(diff)[:topn]
        
        return best[bestcount]
    
    def _pad(self,query,candidate):
        '''
        Ensure that the candidate is at least
        as long as the query. If it isn't, pad
        it with the inverse of the query, so it
        gets the worst possible score for those
        frames
        '''
        if len(candidate) >= len(query):
            return candidate

        diff = len(query) - len(candidate)
        opposite = np.logical_not(query[-diff:])
        return np.concatenate([candidate,opposite])
    
    def _score(self,query,candidate):
        '''
        slide query along candidate, reporting a similarity
        score for each position
        '''
        qlen = len(query)
        scores = np.zeros(1 + (len(candidate) - qlen))
        return [ ((query - candidate[i:i+qlen]) == 0).sum() 
                 for i in xrange(len(scores)) ]
    
    
    def _search(self,frames, nresults):
        # KLUDGE: This algorithm assumes that addresses that are contiguous
        # in the addresses array are contiguous in time.  This is *probably*
        # always a safe assumption, but something about it feels wrong.
        addresses = self.addresses
        index = self.words
        feature = frames[self.feature]
        nblocks = int(len(feature) / self.step)
        ncandidates = nresults
        # keep track of scores in an array where columns represent
        # blocks, and columns represent the n best matches for that block
        scores = np.zeros((nblocks,ncandidates),dtype=np.int32)
        d = {}
        for i,block in enumerate(feature[::self.step]):
            # extract words from this block
            w = np.nonzero(block)[0]
            # get the indices of the top candidates in the addresses array,
            # and set the column corresponding to this block with the top
            # candidates
            if len(w):
                scores[i,:] = self._top_candidates(w, topn = ncandidates)
        
        # sort column-wise, so that we might end up with contiguous addresses
        # together in single rows
        scores = np.sort(scores, axis = 1)
        mx = scores.max(1)
        mn = scores.min(1)
        # measure the "spaciousness" of each row. We want very dense, very
        # non-spacious rows.  this means that the blocks are, or are close
        # to being contigous
        spaciousness = (mx - mn) / scores.shape[1]
        # get sorted row indices, from least to most spacious
        top = np.argsort(spaciousness)
        
        
        env = self.env()
        ac = env.address_class
        fm = env.framemodel
        
        for t in top:
            # "congeal" all the addresses in the most dense row into a single
            # address
            #address = ac.congeal(scores[t])
            # candidates frames
            #cframes = fm[address]
            # get the feature from the candidate frames instance
            #cfeature = cframes[self.feature]
            #cfeature = self._pad(feature,cfeature)
            #sliding_score = self._score(feature,cfeature)
            print addresses[scores[t]]
            print '=================================='
        
            
            
        