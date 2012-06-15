from __future__ import division
from abc import ABCMeta,abstractmethod
import time
import struct
import random
from bisect import bisect_left

import numpy as np
from bitarray import bitarray

from model import Model
from pattern import DataPattern
from nputil import pad,safe_unit_norm as sun
from util import flatten2d
from scipy.spatial.distance import cdist
from environment import Environment

def nbest(query,index,nresults = 10,metric = 'euclidean'):
    dist = cdist(np.array([query]),index,metric)[0]
    best = np.argsort(dist)[:nresults]
    return best, dist[best]

def soundsearch(_ids,index,random = False,nresults = 10, metric = 'euclidean'):
    nsounds = len(_ids)
    indices = np.random.permutation(nsounds) if random else range(nsounds)
    FrameModel = Environment.instance.framemodel
    for i in indices:
        qid = _ids[i]
        best,dist = nbest(index[i],index,nresults = nresults, metric = metric)
        qframes = FrameModel[qid]
        print 'Query %s has length %1.4f' % (qframes.external_id[0],qframes.seconds)
        Environment.instance.play(qframes.audio)
        for i,b in enumerate(best):
            bframes = FrameModel[_ids[b]]
            print 'Sound %s has length %1.4f seconds and distance is %1.4f' % \
                (bframes.external_id[0],bframes.seconds,dist[i])
            Environment.instance.play(bframes.audio)
        raw_input('next...')
        

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
        print l
        r = np.recarray(l,dtype=dtype)
        
        for k,v in d.iteritems():
            if 'audio' == k or\
             (fm.features.has_key(k) and fm.features[k].store):
                rp = np.array(v).repeat(ec[k].step, axis = 0).squeeze()
                padded = pad(rp,l)[:l]
                try:
                    r[k] = padded
                except ValueError:
                    r[k] = flatten2d(padded)
            
        
        # get a frames instance
        frames = fm(data = r)
        return self._search(frames, nresults = nresults)
    
    


class Score(object):
    
    def __init__(self,seq):
        object.__init__(self)
        self.seq = seq
    
    def nbest(self,n):
        b = np.bincount(self.seq)
        nz = np.nonzero(b)[0]
        asrt = np.argsort(b[nz])
        return nz[asrt][::-1][:n]

# TODO: Why is this so much slower now?
class ExhaustiveSearch(FrameSearch):
    
    def __init__(self,_id,feature,step = 40):
        FrameSearch.__init__(self,_id,feature)
        self._std = None
        self._step = step
    
    def _build_index(self):
        env = self.env()
        c = env.framecontroller
        fm = env.framemodel
        _ids = list(fm.list_ids())
        l = len(c)
        
        frames = fm[_ids[0]]
        
        samples = np.zeros((l,frames[self.feature].shape[1]))
        samples[:len(frames)] = frames[self.feature]
        count = len(frames)
        
        for i in range(1,len(_ids)):
            print _ids[i]
            frames = fm[_ids[i]]
            samples[count : count + len(frames)] = frames[self.feature]
        
        self._std = samples.std(0)
        print self._std
    
    @property
    def feature(self):
        return self.features[0]
    
    def _search(self,frames,nresults):
        # get the sequence of query features at the interval
        # specified by self._step
        seq = frames[self.feature][::self._step]
        seq /= self._std
        ls = len(seq)
        seq = seq.ravel()
        
        env = self.env()
        c = env.framecontroller
        _ids = list(c.list_ids())
        # best is a tuple of (score,(_id,addr))
        best = []
        querylen = len(frames)
        for _id in _ids:
            skip = -1
            for addr,frames in c.iter_id(_id,querylen,step = self._step):
                if skip > -1 and skip * self._step < (len(frames) / 2):
                    skip += 1
                    continue
                else:
                    skip = -1
                feat = frames[self.feature]
                feat /= self._std
                feat = pad(feat,ls)
                dist = np.linalg.norm(feat.ravel() - seq)
                t = (dist,(_id,addr))
                try:
                    insertion = bisect_left(best,t)
                except ValueError:
                    print dist
                    print best
                    raise Exception()
                if insertion < nresults:
                    best.insert(insertion,t)
                    best = best[:nresults]
                    if len(best) == nresults:
                        skip = 0
        
        return [t[1] for t in best]
    
#    def _search(self,frames,nresults):
#        seq = frames[self.feature]
#        #seq /= self._std
#        #seq = sun(seq)
#        ls = len(seq)
#        seq = seq.ravel()
#        
#        env = self.env()
#        c = env.framecontroller
#        fm = env.framemodel
#        
#        _ids = list(c.list_ids())
#        # best is a tuple of (score,frames)
#        # KLUDGE: This is cheating. Searches should always return back-end
#        # specific addresses
#        best = []
#        for _id in _ids:
#            frames = fm[_id]
#            feat = frames[self.feature]
#            #feat /= self._std
#            #feat = sun(feat)
#            i = 0
#            print 'comparing to _id %s' % _id
#            while i < len(feat) - ls:
#                
#                sl = feat[i:i+ls]
#                comp = pad(sl,ls)
#                dist = np.linalg.norm(comp.ravel() - seq)
#                t = (dist,(_id,i))
#                
#                try:
#                    insertion = bisect_left(best,t)
#                except ValueError:
#                    print dist
#                    print best
#                    raise Exception()
#                
#                if insertion < nresults and not np.isnan(dist):
#                    best.insert(insertion,t)
#                    best = best[:nresults]
#                    print 'found good match at _id %s, frame %i' % (_id,i)
#                    print t[0]
#                    i += int(ls / 2)
#                else:
#                    i += 1
#        
#        # return just the frames, and not the scores
#        return [fm[t[1][0]].audio[t[1][1] : t[1][1] + ls] for t in best]
            
            
        
class LshSearch(FrameSearch):
    
    _DTYPE_MAPPING = {
                      8  : np.uint8,
                      16 : np.uint16,
                      32 : np.uint32,
                      64 : np.uint64
                      }
    _STRUCT_MAPPING = {
                       8  : 'B',
                       16 : 'H',
                       32 : 'L',
                       64 : 'Q'
                       }
    def __init__(self,_id,feature,step,nbits):
        k = LshSearch._DTYPE_MAPPING.keys()
        if nbits not in k:
            raise ValueError('nbits must be in %s') % (str(k))
        
        FrameSearch.__init__(self,_id,feature)
        self._index = None
        self._sorted = None
        self.step = step
        self.nbits = nbits
        
        self._addresskey = 'a'
        self._hashkey = 'h'
        self._hashdtype = LshSearch._DTYPE_MAPPING[nbits]
        self._structtype = LshSearch._STRUCT_MAPPING[nbits]
        
     
    
    def _bit_permute(self,n):
        '''
        Every possible rotation of n bits
        
        A horribly inefficient way to permute bits. This should be written
        as a wrapped c or cython method
        '''
        # Am I sure this is doing the right thing?
        n = int(n)
        p = np.ndarray(self.nbits,dtype=self._hashdtype)
        for i in range(self.nbits):
            ba = bitarray()
            ba.frombytes(struct.pack(self._structtype,n))
            ba2 = bitarray()
            ba2.extend(np.roll(ba,i))
            p[i] = struct.unpack(self._structtype,ba2.tobytes())[0]
        return p
    
    def _build_index(self):
        '''
        '''
        env = self.env()
        fc = env.framecontroller
        l = len(fc)
        index = np.recarray(\
                l,dtype = [(self._addresskey,np.object),
                           (self._hashkey,self._hashdtype,(self.nbits))])
        
        for i,f in enumerate(fc.iter_all(step = self.step)):
            address,frame = f
            index[i][self._addresskey] = address
            index[i][self._hashkey][:] = self._bit_permute(frame.lsh)
            print index[i][self._hashkey]
        argsort = np.argsort(index[self._hashkey],0)
        self._index = [index,argsort]
    
    @property
    def feature(self):
        return self.features[0]
        
    @property
    def index(self):
        return self._index[0]
    
    @property
    def argsort(self):
        return self._index[1]
    
    @property
    def sorted(self):
        return self._sorted
        
    def _setup(self):
        # TODO: Just save the sorted version of the hash codes. The original
        # version never gets used in the search.
        print 'setting up'
        self._sorted = np.ndarray(\
                    self.index[self._hashkey].shape,dtype = self._hashdtype)
        for i in range(self.nbits):
            self._sorted[:,i] = self.index[self._hashkey][:,i][self.argsort[:,i]]
        print 'done setting up'
        
    
    def _search(self,frames,nresults):
        if None is self._sorted:
            self._setup()
        
        feature = frames[self.feature]
        lf = len(feature)
        env = self.env()
        fm = env.framemodel
        perms = np.zeros((len(feature),self.nbits))
        
        # Get the permutations of the hash code for every block
        for i,f in enumerate(feature):
            perms[i] = self._bit_permute(f)
        rng = 10
        
        l = [[] for i in range(lf)]
        for i in range(self.nbits):
            # get the insertion positions for every block, for this permutation
            inserts = np.searchsorted(self.sorted[:,i],perms[:,i])
            starts = inserts - rng
            starts[starts < 0] = 0
            stops = inserts + rng 
            [l[q].extend(self.argsort[:,i][starts[q] : stops[q]]) for q in range(lf)]
        
        audio = []
        add = audio.append if self.step == 1 else audio.extend
        for candidates in l:
            best = random.choice(Score(candidates).nbest(5))
            a = fm[self.index[self._addresskey][best]].audio
            add(a)
        return np.array(audio)
        
         
        
            
    
           
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
            _id = frames._id
            if isinstance(_id,str):
                ids.append(_id)
            else:
                ids.append(_id[0])
            hsh = frames[self.feature]
            hsh = hsh if 1 == len(hsh.shape) else hsh[0]
            print hsh
            for q,h in enumerate(hsh):
                try:
                    index[q][h].append(i)
                except KeyError:
                    index[q][h] = [i]
                    
        self._index = [np.array(ids),np.array(addresses),index]
    
    
        
    
    def _pad(self,query,candidate):
        '''
        Ensure that the candidate is at least
        as long as the query. If it isn't, pad
        it with the inverse of the query, so it
        gets the worst possible score for those
        frames
        '''
        if 1 == len(candidate.shape):
            candidate = candidate.reshape((1,candidate.shape[0]))
        
        if len(candidate) >= len(query):
            return candidate
        
        querylen = len(query)
        diff = querylen - len(candidate)
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
        return [(query == candidate[i:i+qlen]).sum() 
                    for i in xrange(len(scores))]
    
    
    CACHE = {}
    def _search_block_cached(self,block,candidates):
        tb = tuple(block)
        try:
            return MinHashSearch.CACHE[(tb,candidates)]
        except KeyError:
            val = self._search_block(tb,candidates)
            MinHashSearch.CACHE[(tb,candidates)] = val
            return val
        
    def _search_block(self,hashvalue,nresults):
        index = self.index
        addresses = []
        
        for i,h in enumerate(hashvalue):
            try:
                addresses.extend(index[i][h])
            except KeyError:
                '''
                There are no instances of the (hash function,hash value) pair
                in the database
                '''
                pass
        return Score(addresses).nbest(nresults)
    
    
    def _candidate_sequences(self,feature, candidates_per_block = 50):
        starttime = time.time()
        d = {}
        
        addresses = self.address
        allids = self.ids
        
        
        f = feature[::self.step]
        for block in f:
            # get the n best address indexes that match the query block
            ais = self._search_block_cached(block, candidates_per_block)
            # get the addresses themselves
            addrs = addresses[ais]
            # get the pattern ids that correspond to those blocks
            ids = allids[ais] 
            for i in xrange(len(ids)):
                _id = ids[i]
                addr = addrs[i]
                try:
                    d[_id].add(addr)
                except KeyError:
                    d[_id] = set([addr])
        
        
        env = self.env()
        AC = env.address_class
        candidates = [(_id,AC.congeal(list(addrs))) for _id,addrs in d.iteritems()]
        print '_candidate_sequences took %1.4f' % (time.time() - starttime)
        return candidates
    
    def _score_sequences(self,feature,candidates):
        starttime = time.time()
        # a list that will hold four-tuples of (_id,address,score,pos)
        finalscores = []
        query = feature
        querylen = len(query)
        env = self.env()
        for _id,addr in candidates:
            if len(addr) < querylen * .5:
                continue
            cfeature = env.framemodel[addr][self.feature]
            cfeature = self._pad(query,cfeature)
            scores = self._score(query,cfeature)
            [finalscores.append((_id,addr,s,i)) for i,s in enumerate(scores)]
        
        finalscores.sort(key = lambda fs : fs[2], reverse = True)
        print '_score_sequences took %1.4f' % (time.time() - starttime)
        return finalscores
    
    def _avoid_overlap(self,nresults,finalscores,querylen):
        starttime = time.time()
        tolerance = querylen * .85
        AC = self.env().address_class
        finalresults = []
        allstarts = []
        count = 0
        # avoid results that overlap with previous results too much
        while len(finalresults) < nresults and count < len(finalscores):
            _id,addr,score,pos = finalscores[count]
            # KLUDGE: This is cheating. I'm using knowledge about the frames back-end
            # implementation here, which is a no-no!/
            
            # BUG: This approach makes it possible for results to span multiple\
            # patterns
            start = addr.key.start + pos
            stop = start + querylen
            if not np.any(np.array([abs(start - z) for z in allstarts]) <= tolerance):
                finalresults.append((_id,AC(slice(start,stop))))
                allstarts.append(start)
            count += 1
        
        print '_avoid_overlap took %1.4f' % (time.time() - starttime)
        return finalresults
    
    def _search(self,frames, nresults):
        # TODO:
        # some results are spanning multiple patterns
        # Lots of near duplicate clips from the same sound
        # Remove any knowledge of frames back-end
        # Search is slooow
        feature = frames[self.feature]
        candidates = self._candidate_sequences(feature)
        finalscores = self._score_sequences(feature, candidates)
        querylen = len(feature)
        return self._avoid_overlap(nresults,finalscores, querylen)
        

            
        