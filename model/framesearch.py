from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

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

class Score(dict):
    
    def __init__(self,seq):
        dict.__init__(self)
        self.seq = seq
        self._score()
    
    def _score(self):
        for s in self.seq:
            try:
                self[s] += 1
            except KeyError:
                self[s] = 1
        
    def nbest(self,n):
        items = self.items()
        # sort from highest to lowest value
        items.sort(key = lambda i : i[1], reverse = True)
        # return the best ranked keys
        return [i[0] for i in items[:n]]
        
                
class MinHashSearch(FrameSearch):
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
            value1 : [blocks.....],
            value2 : [blocks.....]
        }
        // has function 1
        {
            value : [blocks.....]
        }
        ...
    ]
    '''
    def __init__(self,_id,feature,step,size):
        FrameSearch.__init__(self,_id,feature)
        self._index = None
        self.step = step
        self.size = size
    
    @property
    def feature(self):
        return self.features[0]
    
    @property
    def ids(self):
        return self._index[0]

    @property
    def address(self):
        return self._index[1]

    @property
    def index(self):
        return self._index[2]
    
    def _build_index(self):
        env = self.env()
        fc = env.framecontroller
        addresses = []
        ids = []
        index = [{} for s in xrange(self.size)]
        for i,f in enumerate(fc.iter_all(step = self.step)):
            address,frames = f
            addresses.append(address)
            ids.append(frames._id[0])
            hsh = frames[self.feature]
            hsh = hsh if 1 == len(hsh.shape) else hsh[0]
            print hsh
            for q,h in enumerate(hsh):
                try:
                    index[q][h].append(i)
                except KeyError:
                    index[q][h] = [i]
                    
        self._index = [np.array(ids),np.array(addresses),index]
    
    def _search_block(self,hashvalue,nresults):
        index = self.index
        addresses = []
        for i,h in enumerate(hashvalue):
            try:
                addresses.extend(index[i][h])
            except KeyError:
                '''
                There are no instances of the (hash function,has value) pair
                in the database
                '''
                pass
        return Score(addresses).nbest(nresults)
        
    
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
        opposite = np.ndarray((diff,query.shape[1]))
        opposite[:] = -1
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
        # TODO:
        # finalresults aren't being sorted by score
        # Break this method up into smaller pieces
        # MemoryErrors.  Where are these huge slices coming from
        # Lots of near duplicate clips from the same sound
        # Search is slooow
        feature = frames[self.feature]
        addresses = self.address
        allids = self.ids
        
        d = {}
        for block in feature[::self.step]:
            # get the n best address indexes that match the query block
            ais = self._search_block(block, 100)
            # get the addresses themselves
            addrs = [addresses[ai] for ai in ais]
            # get the pattern ids that correspond to those blocks
            ids = [allids[ai] for ai in ais]
            for i in xrange(len(ids)):
                _id = ids[i]
                addr = addrs[i]
                if not d.has_key(_id):
                    d[_id] = [addr]
                elif addr not in d[_id]:
                    d[_id].append(addr)
        
        items = d.items()
        # KLUDGE: This will prefer longer sounds to better matches
        items.sort(key = lambda item : len(item[1]), reverse = True)
        env = self.env()
        AC = env.address_class
        candidates = [(_id,AC.congeal(addrs)) for _id,addrs in items[:nresults * 2]]
        
        # a list that will hold four-tuples of (_id,address,score,pos)
        finalscores = []
        query = feature
        querylen = len(query)
        tolerance = querylen * .85
        for _id,addr in candidates:
            print 'reading feature'
            cfeature = env.framemodel[addr][self.feature]
            print 'read feature'
            print 'padding' 
            cfeature = self._pad(query,cfeature)
            print 'padded'
            print 'scoring'
            scores = self._score(query,cfeature)
            print 'scored'
            
            for i,s in enumerate(scores):
                finalscores.append((_id,addr,s,i))
        
        finalscores = sorted(finalscores, key = lambda fs : fs[2], reverse = True)
        
        finalresults = []
        allstarts = []
        count = i
        while len(finalresults) < nresults and count < len(finalscores):
            _id,addr,score,pos = finalscores[i]
            # KLUDGE: This is cheating. I'm using knowledge about the frames back-end
            # implementation here, which is a no-no!/
            start = addr.key.start + pos
            stop = start + querylen
            # avoid results that overlap with previous results too much
            print score / (query.size)
            if not np.any(np.array([abs(start - z) for z in allstarts]) <= tolerance):
                finalresults.append((_id,AC(slice(start,stop))))
                allstarts.append(start)
            i += 1
            
        
        
        return finalresults
            
        
        
            
            
        
        
        
        
            
            
        