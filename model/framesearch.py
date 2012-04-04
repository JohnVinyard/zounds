from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

from model import Model
from pattern import DataPattern
from util import pad

# TODO: I'm not sure the model module package is the appropriate place for this,
# and/or if it should be Model-derived
class FrameSearch(Model):
    '''
    FrameSearch-derived classes use the frames backing store to provide results
    to queries in the form of sound using one or more stored features.
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self,*features):
        Model.__init__(self)
        self.features = features
    
    @abstractmethod
    def _build_index(self):
        '''
        Build any data structures needed by this search and persist them 
        somehow
        '''
        pass
    
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

# start simple. Build an index that maps words -> (pattern_id,density). Score
# sounds by balancing number of words in common with word density
class BinaryFeatureSearch(FrameSearch):
    
    def __init__(self,feature):
        FrameSearch.__init__(self,feature)
        self._index = None
    
    @property
    def feature(self):
        return self.features[0]
    
    def _extract_data(self,frames):
        '''
        Extract the "words" present (non-zero items), as well as the density
        of words
        '''
        f = frames[self.feature]
        s = f.sum(0)
        density = s.sum() / f.size
        return np.nonzero(s)[0], density
    
    def _build_index(self):
        self._index = {}
        env = self.env()
        framemodel = env.framemodel
        _ids = framemodel.list_ids()
        for _id in _ids:
            frames = framemodel[_id]
            words,density = self._extract_data(frames)
            print '_id : %s, nwords : %i, density : %1.4f' %\
                     (_id,len(words),density)
            nwords = len(words)
            for w in words:
                t = (_id,nwords)
                try:
                    self._index[w].append(t)
                except KeyError:
                    self._index[w] = [t]
    
    
    def _search(self,frames, nresults):
        print frames.rbm
        words,density = self._extract_data(frames)
        results = {}
        counts = {}
        nwords = len(words)
        for w in words:
            try:
                has_word = self._index[w]
                for _id,count in has_word:
                    inc = 1
                    counts[_id] = count
                    try:
                        results[_id] += inc 
                    except KeyError:
                        results[_id] = inc
            except KeyError:
                # this word wasn't in any pattern used to build the index
                pass
        
        items = results.items()
        items = sorted(items,key=lambda i : i[1],reverse=True)
        item_ids = [i[0] for i in items]
        
        c = counts.items()
        c = sorted(items, key=lambda i : i[1])
        c_ids = [c[0] for c in items]
        
        
        # TODO: This needs to be returning a list of backend specific addresses!
        #return [k for k,v in items[:nresults]]
        combined = list(set(item_ids[:20]) & set(c_ids[:20]))
        items = filter(lambda item : item[0] in combined,items)
        return [item[0] for item in items[:nresults]]
        