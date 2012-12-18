'''

One of zounds' primary motivating factors is to develop the building blocks for 
a fast, high-quality audio similarity search.  The 
:py:mod:`zounds.model.framesearch` module contains classes which define a 
common API for search implementations, and a couple concrete implementations.

In general, a search should use one or more pre-computed features to index the
frames database.

Concretely, assume this is the current 
:py:class:`~zounds.model.frame.Frames`-derived class::
    
    class FrameModel(Frames):
        fft = Feature(FFT, store = False)
        bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
        centroid = Feature(SpectralCentroid, needs = bark, store = False)
        flatness = Feature(SpectralFlatness, needs = bark, store = False)
        vec = Feature(Composite, needs = [centroid,flatness])


Here's how you'd perform an :py:class:`ExhaustiveSearch` using the composite
spectral centroid and spectral flatness feature called :code:`vec`::

    >>> search = ExhaustiveSearch('search/mysearch',FrameModel.vec,normalize = True)
    >>> search
    ExhaustiveSearch(
        normalize = True,
        step = 1,
        do_max = False,
        feature = Feature(extractor_cls = Composite, store = True, key = vec))
    >>> search.build_index()
    >>> frames = FrameModel.random()[:60] # take the first 60 frames of a random sound
    >>> search.search(frames)
    [('408ca7fcffd2492c89a59d8f1c685a30', <class 'zounds.data.frame.pytables.Address'> - slice(3487, 3546, None)), 
    ('e5a3f19dd45d459baab8f3a3f8746576', <class 'zounds.data.frame.pytables.Address'> - slice(3282, 3341, None)), 
    ('e5a3f19dd45d459baab8f3a3f8746576', <class 'zounds.data.frame.pytables.Address'> - slice(3322, 3381, None))]

The results are returned as a list of two-tuples of 
:code:`(zounds_id,backend-specific address)`, sorted in order of descending 
relevance to your query.  :py:meth:`FrameSearch.search` can also accept a
backend-specifc address, or raw audio samples in the form of a numpy array.

If you used the :doc:`quickstart script <quick-start>` to start your application,
there's a handy command-line tool, :code:`search.py`, which handles many search
details for you.  It will also play the search results, so you can judge the
quality of the search results with your own ears!  Here's how you'd use the
command-line tool to perform the same search we just did in code::

    python search.py --feature vec --searchclass ExhaustiveSearch --sounddir sound --nresults 3 --normalize True

If you run... ::
    
    python search.py --help

... you'll notice that :code:`normalize` isn't one of the options listed there.
You can pass arbitrary additional arguments to :code:`search.py`, and they'll
be passed along to the constructor of whichever search class you're using.

'''

from __future__ import division
from abc import ABCMeta,abstractmethod
from time import time,sleep
import struct
from bisect import bisect_left
import os
from multiprocessing import Pool
import logging
from threading import Thread

import numpy as np
from scipy.spatial.distance import cdist
from bitarray import bitarray

from model import Model
from pattern import DataPattern
from zounds.nputil import \
    hamming_distance,pad,Packer,packed_hamming_distance,TypeCodes,flatten2d,Growable
from zounds.environment import Environment
from zounds.util import tostring

LOGGER = logging.getLogger(__name__)

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
        

class Persistor(Thread):
        
    def __init__(self,framesearch,persist_every = 10):
        Thread.__init__(self)
        self.daemon = True
        self.framesearch = framesearch
        self.persist_every = persist_every
        self._should_stop = False
    
    def update(self):
        del self.framesearch.__class__[self.framesearch._id]
        self.framesearch.store()
    
    def run(self):
        while not self._should_stop:
            # TODO: I should be using a thread and process safe lock here
            self.update()
            sleep(self.persist_every * 60)
            
    def stop(self):
        self._should_stop = True
        self.update()


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
    Framesearch is an abstract base class.  The 
    :py:meth:`FrameSearch._build_index` and 
    :py:meth:`FrameSearch._search` methods must be implemented by inheriting 
    classes.
    
    FrameSearch-derived classes use the frames backing store to provide results
    to queries in the form of sound or precomputed features using one or more 
    stored features.
    '''
    __metaclass__ = MetaFrameSearch
    
    
    def __init__(self,_id,*features):
        '''__init__
        
        :param _id: The _id that will be used to persist and fetch this search
        
        :param features: One or more \
        :py:class:`~zounds.model.frame.Feature`-derived instances that will be \
        used by the search to determine similarity or relevance.
        
        '''
        Model.__init__(self)
        self._id = _id
        self.features = features
    
    def store(self):
        self.controller().store(self)
    
    def __repr__(self):
        return tostring(self,id = self._id,feature = self.features)
    
    def __str__(self):
        return self.__repr__()

    def start_persistor(self):
        self.persistor.start()
    
    def stop_persistor(self):
        self.persistor.stop()
    
    #TODO: Is this method necessary? Isn't this defined on the MetaFrameSearch
    # class?
    @classmethod
    def __getitem__(cls,key):
        return cls.controller()[key]
    
    @abstractmethod
    def _build_index(self):
        '''
        **Must be implemented by inheriting classes**
        
        Build any data structures needed by this search
        '''
        pass

    @abstractmethod
    def _add_index(self,_id):
        '''
        Add a single pattern to the index
        '''
        pass
    
    @abstractmethod
    def _check_index(self):
        '''
        Ensure that the index is in sync with the Frames database
        '''
        pass
    
    def _add_id(self,_id):
        '''
        Add _id to the index
        '''
        raise NotImplemented()
    
    def build_index(self):
        '''
        Build data structures need for this search and persist them for future
        use
        '''
        self._build_index()
        self.store()
    
    def add_index(self,_id):
        self._add_index(_id)
    
    def check_index(self):
        self._check_index()
        del self.controller()[self._id]
        self.store()
    
    @abstractmethod
    def _search(self,frames):
        '''
        **Must be implemented by inheriting classes**
        
        Perform the search, returning a list of two-tuples of 
        :code:`(zounds_id,address)`, where :code:`address` is an instance
        of :py:attr:`~zounds.environment.Environment.address_class`.
        
        :param frames: a :py:class:`~zounds.model.frame.Frames`-derived instance
        '''
        pass
    
   
    # KLUDGE: This is a total mess!  Doing on the fly audio extraction should
    # be much easier and nicer than this.
    
    # TODO: Frames should have an extract method, which transforms raw audio
    # into a frames instance which is identical to one that would be returned
    # from the database. 
    def search(self,query, nresults = 10):
        '''
        Perform a search
        
        :param query: query can be one of
            
            * a :py:class:`~zounds.model.frame.Frames`-derived instance
            * an address, which points at frames in the datastore
            * a numpy array containing raw audio samples. The current set of
              features will be computed for the query audio.
        
        :returns: :code:`(frames,results)` where :code:`frames` is the \
        :py:class:`~zounds.model.frame.Frames`-derived instance that was either \
        passed in, fetched from the database, or computed, and :code:`results` \
        is a list of two-tuples of :code:`(zounds_id,address)` where \
        :code:`address` is an instance of the current \
        :py:attr:`~zounds.environment.Environment.address_class`.
        '''
        env = self.env()
        
        start = time()
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
        
        audio = np.concatenate(d['audio'])
        l = len(audio)
        r = np.recarray(l,dtype=dtype)
        
        for k,v in d.iteritems():
            if 'audio' == k or\
             (fm.features.has_key(k) and fm.features[k].store):
                data = np.concatenate(v)
                rp = data.repeat(ec[k].step_abs(), axis = 0).squeeze()
                padded = pad(rp,l)[:l]
                try:
                    r[k] = padded
                except ValueError:
                    r[k] = flatten2d(padded)
        
        # get a frames instance
        frames = fm(data = r)
        stop = time() - start
        LOGGER.info('analysis took %1.4f seconds' % stop)
        return frames,self._search(frames, nresults = nresults)
    
    


class Score(object):
    
    def __init__(self,seq):
        object.__init__(self)
        self.seq = seq
    
    def nbest(self,n):
        b = np.bincount(self.seq)
        nz = np.nonzero(b)[0]
        asrt = np.argsort(b[nz])
        # Get the top n occurrences, in descending order of frequency
        return nz[asrt][-n:][::-1]
    
    

# KLUDGE ###################################################################
# The following code is used by the parallel search implementation of 
# ExhaustiveSearch.  It really sucks that instance and class methods can't
# be pickled, and thus can't be easily used by the multiprocessing module.
# Perhaps I should move all things ExhautiveSearch into a seperate module, 
# at least

LOCK_NAME = 'ExhaustiveSearch.lock'

def acquire_lock():
    while os.path.exists(LOCK_NAME):
        time.sleep(0.1)
    with open(LOCK_NAME,'w') as f:
        pass
    
def release_lock():
    try:
        os.remove(LOCK_NAME)
    except OSError:
        pass

def _search_parallel(args):
    _ids,nresults,key,feature,step,std = args
    ls = len(feature)
    feature /= std
    seq = feature.ravel()
    
    # KLUDGE: Opening handles to the same PyTables file from multiple processes
    # at the same time seems to cause all kinds of unpredictable craziness.
    # Once we're passed this point, concurrent reads seem to work OK.  Force
    # the opening of the file to happen serially.
    acquire_lock()
    c = Environment.instance.framecontroller_class(\
                    *Environment.instance._framecontroller_args)
    release_lock()
    print 'GOT PYTABLES HANDLE'
    best = []
    querylen = len(feature)
    for _id in _ids:
        skip = -1
        for addr,frames in c.iter_id(_id,querylen,step = step):
            if skip > -1 and skip * step < (ls / 2):
                skip += 1
                continue
            else:
                skip = -1
            feat = frames[key]
            feat /= std
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
    return best

############################################################################

class ExhaustiveSearch(FrameSearch):
    '''
    Find similar segments of sound by taking the euclidean distance between
    the query's features and stored features at every valid position in the 
    database.
    
    This brute force approach is not appropriate for large databases, but can be
    used on smaller sets of sounds to evaluate the performance of a certain feature.
    '''
    
    def __init__(self,_id,feature,step = 1,
                 normalize = True,multi_process = False,do_max = False):
        
        '''__init__
        
        :param _id:  The key that will be used to store and retrieve this instance
        
        :param feature: a :py:class:`~zounds.model.frame.Feature` instance that \
        is currently stored
        
        :param step: The interval at which frames from the query and equal-length \
        spans of frames from the database should be compared.  Typically, this \
        will be the absolute step value of the feature.
        
        :param normalize: If :code:`True`, and :code:`feature` is multi-dimensional, \
        all feature values from the query and the database will be divided \
        feature-wise by the feature's standard deviation, so that all dimensions \
        of the feature are given equal variance.
         
        '''
        
        FrameSearch.__init__(self,_id,feature)
        self._std = None
        self._step = step
        self._normalize = normalize
        self._multi_process = multi_process
        self._do_max = do_max
    
    def __repr__(self):
        return tostring(self,short = False,feature = self.feature,step = self._step,
                        normalize = self._normalize, do_max = self._do_max)
    
    def __str__(self):
        return tostring(self,feature = self.feature,step = self._step)
    
    def _search(self,frames,nresults):
        if self._multi_process:
            return self._search_multi_process(frames, nresults)
        
        return self._search_single_process(frames, nresults)
        
    
    def _build_index(self):
        if self._normalize:
            self._std = self.feature.std()
            print self._std
        else:
            self._std = 1
    
    def _add_index(self,_id):
        # TODO: update self._std
        raise NotImplemented()
    
    def _check_index(self):
        '''
        This is a no-op, since this search stores no index data
        '''
        pass
    
    @property
    def feature(self):
        return self.features[0]
    
    
    def _search_multi_process(self,frames,nresults):
        '''
        Split the _id space into four chunks, and search them seperately.
        Combine the results from each search to find the nresults best segments.
        '''
        # make sure the lock is released, just in case something went
        # wrong last time
        release_lock()
        nprocesses = 4
        _ids = list(self.env().framecontroller.list_ids())
        chunksize = int(len(_ids) / nprocesses)
        chunks = []
        for i in xrange(nprocesses):
            start = i*chunksize
            stop = start + chunksize
            chunks.append(_ids[start:stop])
        seq = frames[self.feature][::self._step]
        key = self.feature.key
        args = [(c,nresults,key,seq,self._step,self._std) \
                for c in chunks]
        p = Pool(processes = nprocesses)
        scores = p.map(_search_parallel,args)
        final = self._combine(nresults,*scores)
        return final
    
    
    def _combine(self,nresults,*scores):
        '''
        Combine results from multiple _search processes
        '''
        s = []
        [s.extend(score) for score in scores]
        s.sort()
        return [t[1] for t in s[:nresults]]
    
    
    def _search_single_process(self,frames,nresults):
        # get the sequence of query features at the interval
        # specified by self._step
        seq = frames[self.feature][::self._step]
        
        if self._normalize:
            seq /= self._std
        ls = len(seq)
        
        if not self._do_max:
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
                if skip > -1 and skip * self._step < (querylen / 2):
                    skip += 1
                    continue
                else:
                    skip = -1
                feat = frames[self.feature]
                
                if self._do_max:
                    feat /= self._std
                    feat = feat.max(0)
                    seq2 = seq.max(0)
                    dist = np.linalg.norm(feat - seq2)
                else:
                    feat /= self._std
                    feat[np.isnan(feat)] = 0
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


class Frequency(object):
    
    def __init__(self):
        object.__init__(self)
        self._freq = dict()
    
    def count(self,v):
        try:
            self._freq[v] += 1
        except KeyError:
            self._freq[v] = 1
    
    def weights(self,arr):
        w = np.ndarray(arr.shape)
        for i,a in enumerate(arr):
            w[i] = self._freq[a]
        return w / len(arr)


class ExhaustiveLshSearch(FrameSearch):
    '''
    '''
    def __init__(self,_id,feature,step = None,fine_feature = None,
                 ignore = None, growth_rate = .25,initial_size = None):
        
        # 76561193665298448
        # 100010000000000000000000000000000000000000000000000010000
        
        # 8001096117072440285
        
        FrameSearch.__init__(self,_id,feature)
        self._hashdtype = feature.dtype
        self.step = step
        self.nbits = TypeCodes.bits(self._hashdtype)
        self._initial_size = initial_size
        self._growth_rate = growth_rate
        
        # allocate enough memory (plus a little) to hold an index for the
        # entire database as it currently exists
        self._index = self._allocate(self._hashdtype)
        self._fine_feature = fine_feature
        self._fine_index = None
        if self._fine_feature:
            fc = self.env().framecontroller
            dim = fc.get_dim(self._fine_feature)
            self._packer = Packer(dim[0])
            self._fine_feature = self._allocate_fine()
            
        self._addrs = self._allocate(object)
        self._ids = set()
        self._logical_size = 0
        
        
        # blocks in queries with any of the values in self._filter will be
        # ignored when performing the search
        self._filter = set([0])
        if None is not ignore:
            try:
                self._filter.update(ignore)
            except TypeError:
                self._filter.add(ignore)
        
        
    def __repr__(self):
        return tostring(self,short = False,feature = self.feature,id = self._id,
                        step = self.step, nbits = self.nbits, 
                        fine_feature = self._fine_feature,ignore = self._filter)
    def __str__(self):
        return tostring(self,id = self._id,feature = self.feature,
                        step = self.step,nbits= self.nbits)
    
    @property
    def feature(self):
        return self.features[0]
    
    @property
    def framecontroller(self):
        return self.env().framecontroller
    
    def _feature_value(self,frame,feature):
        try:
            return frame[feature][0]
        except IndexError:
            return frame[feature]
    
    def _safe_size(self):
        if None is self._initial_size:
            fc = self.framecontroller
            minimum = np.ceil(len(fc) / self.step)
        else:
            minimum = self._initial_size
        return minimum + int(minimum * self._growth_rate)
    
    def _allocate(self,dtype):
        return np.ndarray(self._safe_size(),dtype = dtype)
    
    def _allocate_fine(self):
        size = self._safe_size()
        return self._packer.allocate(size)
    
    def _add_index(self,_id):
        '''
        Add frames to the existing index. Keep a temporary index, and don't
        write the frames until building is complete. Update self._ids when
        done
        '''
        env = self.env()
        fc = self.framecontroller
        l = np.ceil(fc.pattern_length(_id) / self.step)
        new_index = np.ndarray(l,self._hashdtype)
        if self._fine_feature:
            # KLUDGE: This is duplicated exactly in _build_index()
            dim = fc.get_dim(self._fine_feature)
            self._packer = Packer(dim[0])
            new_fine_index = self._packer.allocate(l)
        
        new_addrs = np.ndarray(l,object)
        chunksize = env.chunksize_frames
        index = 0
        
        for address,frame in fc.iter_id(_id,chunksize,step = self.step):
            fv = self._feature_value(frame, self.feature)
            if fv not in self._filter:
                new_addrs[index] = (_id,address)
                new_index[index] = fv
                
                if self._fine_feature:
                    # add the fine feature value to the fine feature index
                    ffv = self._feature_value(frame, self._fine_feature)
                    if not ffv.shape:
                        new_fine_index[index] = 0
                    else:
                        new_fine_index[index] = self._packer(ffv[np.newaxis,...])
                fv
                index += 1
        
        
        if self._fine_feature:
            new_fine_index = new_fine_index[:index + 1]
            self._fine_index = Growable(self._fine_index, 
                                        position = self._logical_size, 
                                        growth_rate = self._growth_rate) \
                                .extend(new_fine_index).data
        
        new_addrs = new_addrs[:index + 1]
        self._addrs = Growable(self._addrs,
                               position = self._logical_size,
                               growth_rate = self._growth_rate) \
                               .extend(new_addrs).data
        
        # update the main index last, as this is the first index consulted by
        # calls to _search.  This should mostly avoid problems caused by
        # searches using out-of-sync indices.
        new_index = new_index[:index + 1]
        #self._index = np.concatenate(self._index,new_index)
        new_index = Growable(self._index,
                               position = self._logical_size,
                               growth_rate = self._growth_rate) \
                               .extend(new_index)
        self._logical_size =  new_index.logical_size
        self._index = new_index.data
        self._ids.add(_id)
         
    
    def _check_index(self):
        # check FrameModel.list_ids() against self.list_ids() and do any work
        # necessary to update the index 
        actual_ids = self.framecontroller.list_ids()
        diff = actual_ids ^ self._ids
        for _id in diff:
            self._add_index(_id)
        
    
    # KLUDGE: It's possible to build a corrupted index using this method.  More
    # specifically, it's possible to add entries to the index array for some 
    # pattern id, and fail before the pattern has been fully processed, leaving
    # the pattern id out of self._ids.  When _check_id is called, this id
    # will be missing, and will be added to the index again, meaning that there
    # will be some duplicated entries.
    #
    # Using _add_index for each _id is actually more robust, but less efficient,
    # because new memory is allocated for each new pattern.  One solution might
    # be to *always* use _add_index, and keep some temporary memory allocated
    # at all times, instead of allocating memory and throwing it away with each
    # _add_index call. 
    def _build_index(self):
        fc = self.env().framecontroller
        index = Growable(self._index, position = self._logical_size, 
                         growth_rate = self._growth_rate)
        addrs = Growable(self._addrs, position = self._logical_size,
                         growth_rate = self._growth_rate)
        
        print self._fine_feature
        if self._fine_feature:
            findex = Growable(self._index, position = self._logical_size,
                              growth_rate = self._growth_rate)
        
        last_id = None
        for address,frame in fc.iter_all(step = self.step):
            fv = self._feature_value(frame,self.feature)
            _id = frame['_id'][0]
            # Note that we're keeping track of ids, regardless of whether any
            # part of this pattern makes it into the index.  This indicates
            # that we've processed this pattern in its entirety, and it doesn't
            # need to be re-indexed.  Note that the _id only gets added once
            # the entire pattern has been processed.
            if last_id is None:
                last_id = _id
            elif _id != last_id:
                self._ids.add(last_id)
            
            if fv not in self._filter:
                # only include codes that aren't included in self._filter.
                # Similarity at those positions doesn't matter.
                addrs.append((_id,address))
                index.append(fv)
                
                if self._fine_feature:
                    # add the fine feature value to the fine feature index
                    ffv = self._feature_value(frame, self._fine_feature)
                    if not ffv.shape:
                        findex.append(0)
                    else:
                        findex.append(self._packer(ffv[np.newaxis,...]))
                        
                print fv
                self._logical_size = index.logical_size
        
        if last_id is not None:
            self._ids.add(last_id)
        
        self._index = index.data
        self._addrs = addrs.data
        if self._fine_feature:
            self._fine_index = findex.data
    
    def _valid_indices(self,features):
        s = np.sum([(features == q) for q in self._filter],0)
        return s == 0 
        
    def _search(self,frames,nresults):
        start = time()
        feature = frames[self.feature][::self.step]
        valid = self._valid_indices(feature)
        some_valid = np.any(valid)
        # If any of the frames are valid, remove invalid frames, otherwise,
        # we have no choice but to use everything.
        feature = feature[valid] if some_valid else feature
        
        if self._fine_feature:
            ff = frames[self._fine_feature][::self.step]
            ff = ff[valid] if some_valid else ff
            ff = self._packer(ff)
        
        lf = len(feature)
        ls = self._logical_size
        # KLUDGE: Results may not be unique
        indices = []
        distances = []
        for i in range(lf):
            dist = hamming_distance(feature[i],self._index[:ls])
            srt = np.argsort(dist)
            
            # TODO: Make this size configurable
            # TODO: What effect does altering this size have?
            n = nresults * 10
            
            if self._fine_feature:
                finer = packed_hamming_distance(ff[i],self._fine_index[:ls][srt[:n]])
                fsrt = np.argsort(finer)
                srt = srt[:n][fsrt]
                dist = dist[srt]
                indices.extend(srt)
                distances.extend(finer[fsrt])
            else:
                indices.extend(srt[:n])
                distances.extend(dist[srt[:n]])
        
        
        dsrt = np.argsort(distances)
        indices = np.array(indices)[dsrt[:nresults]]
        results = [addr for addr in self._addrs[:ls][indices]]
        
            
        stop = time() - start
        print 'search took %1.4f seconds' % stop
        return results

#class ExhaustiveLshSearch(FrameSearch):
#    '''
#    Quickly search large databases using features which are stored as
#    32 or 64 bit scalars.  The scalars are treated as binary feature vectors
#    of dimension 32 or 64, and are compared using the hamming distance.
#    
#    Works well for features computed using 
#    `locality-sensitive hashing <http://en.wikipedia.org/wiki/Locality-sensitive_hashing>`_
#    or 
#    `semantic hashing <http://www.utstat.toronto.edu/~rsalakhu/papers/semantic_final.pdf>`_
#    '''
#    
#    # TODO: nbits could be inferred from the feature
#    def __init__(self,_id,feature,step = None,nbits = None,
#                 fine_feature = None,ignore = None):
#        
#        # 76561193665298448
#        # 100010000000000000000000000000000000000000000000000010000
#        
#        # 8001096117072440285
#        
#        '''__init__
#        
#        :param _id:  The key that will be used to store and retrieve this instance
#        
#        :param feature: a :py:class:`~zounds.model.frame.Feature` instance that \
#        is currently stored
#        
#        :param step: The interval at which frames from the query and equal-length \
#        spans of frames from the database should be compared.  Frequently, this \
#        will be the absolute step value of the feature.
#        
#        :param nbits: 32 or 64
#        
#        :param fine_feature: Please just ignore this for now
#        
#        :param ignore: a list of 32 or 64 bit unsigned integers which represent \
#        codes that should be ignored, i.e., that should not figure into sequence \
#        similarity either negatively or positively
#    
#        '''
#        
#        k = TypeCodes._bits
#        if nbits not in k:
#            raise ValueError('nbits must be in %s' % (str(k)))
#        
#        FrameSearch.__init__(self,_id,feature)
#        self._index = None
#        self._fine_feature = fine_feature
#        self._fine_index = None
#        self._addrs = None
#        # blocks in queries with any of the values in self._filter will be
#        # ignored when performing the search
#        self._filter = [0]
#        if None is not ignore:
#            try:
#                self._filter.extend(ignore)
#            except TypeError:
#                self._filter.append(ignore)
#        self.step = step
#        self.nbits = nbits
#        self._hashdtype = TypeCodes.np_dtype(nbits)
#        self._structtype = TypeCodes.type_code(nbits)
#        
#    
#    def __repr__(self):
#        return tostring(self,short = False,feature = self.feature,id = self._id,
#                        step = self.step, nbits = self.nbits, 
#                        fine_feature = self._fine_feature,ignore = self._filter)
#    def __str__(self):
#        return tostring(self,id = self._id,feature = self.feature,
#                        step = self.step,nbits= self.nbits)
#    
#    @property
#    def feature(self):
#        return self.features[0]
#    
#    def _feature_value(self,frame,feature):
#        try:
#            return frame[feature][0]
#        except IndexError:
#            return frame[feature]
#    
#    def _check_index(self):
#        raise NotImplemented()
#    
#    # TODO: There's a ton of duplicated logic in self._build_index. Factor some
#    # of it out.
#    def _add_index(self,_id):
#        # allocate enough memory for an index over just this _id
#        # iterate over the frames, populating the single id index
#        # concatenate the single _id index with the existing one
#        # swap out the indexes
#        env = self.env()
#        fc = env.framecontroller
#        l = np.ceil(fc.pattern_length(_id) / self.step)
#        new_index = np.ndarray(l,self._hashdtype)
#        if self._fine_feature:
#            # KLUDGE: This is duplicated exactly in _build_index()
#            dim = fc.get_dim(self._fine_feature)
#            self._packer = Packer(dim[0])
#            new_fine_index = self._packer.allocate(l)
#        
#        new_addrs = np.ndarray(l,object)
#        chunksize = env.chunksize_frames
#        index = 0
#        
#        for address,frame in fc.iter_id(_id,chunksize,step = self.step):
#            fv = self._feature_value(frame, self.feature)
#            if fv not in self._filter:
#                new_addrs[index] = (_id,address)
#                new_index[index] = fv
#                
#                if self._fine_feature:
#                    # add the fine feature value to the fine feature index
#                    ffv = self._feature_value(frame, self._fine_feature)
#                    if not ffv.shape:
#                        new_fine_index[index] = 0
#                    else:
#                        new_fine_index[index] = self._packer(ffv[np.newaxis,...])
#                print new_index[index]
#                index += 1
#        
#        
#        # lop off any unused indices
#        if self._fine_feature:
#            new_fine_index = new_fine_index[:index + 1]
#            self._fine_index = np.concatenate(self._fine_index,new_fine_index)
#        
#        new_addrs = new_addrs[:index + 1]
#        self._addrs = np.concatenate(self._addrs,new_addrs)
#        
#        # update the main index last, as this is the first index consulted by
#        # calls to _search.  This should mostly avoid problems caused by
#        # searches using out-of-sync indices.
#        new_index = new_index[:index + 1]
#        self._index = np.concatenate(self._index,new_index)
#        
#    
#    def _build_index(self):
#        env = self.env()
#        fc = env.framecontroller
#        l = int(len(fc) / self.step)
#        
#        # get the id for every frame instance
#        nids = len(fc.list_ids())
#        # An index that will hold the primary binary feature
#        self._index = np.ndarray(l+nids,self._hashdtype)
#        if self._fine_feature:
#            # allocate enough memory to "pack" the boolean numpy array that
#            # represents the fine feature into bits
#            dim = fc.get_dim(self._fine_feature)
#            self._packer = Packer(dim[0])
#            self._fine_index = self._packer.allocate(l + nids)
#        
#        self._addrs = np.ndarray(l+nids,object)
#        index = 0
#        for address,frame in fc.iter_all(step = self.step):
#            fv = self._feature_value(frame,self.feature)
#            
#            if fv not in self._filter:
#                # only include codes that aren't included in self._filter.
#                # Similarity at those positions doesn't matter.
#                _id = frame['_id'][0]
#                self._addrs[index] = (_id,address)
#                self._index[index] = fv
#                
#                if self._fine_feature:
#                    # add the fine feature value to the fine feature index
#                    ffv = self._feature_value(frame, self._fine_feature)
#                    if not ffv.shape:
#                        self._fine_index[index] = 0
#                    else:
#                        self._fine_index[index] = self._packer(ffv[np.newaxis,...])
#                print self._index[index]
#                index += 1
#        
#        # lop off any unused indices
#        if self._fine_feature:
#            self._fine_index = self._fine_index[:index + 1]
#        self._index = self._index[:index + 1]
#        self._addrs = self._addrs[:index + 1]
#        
#    
#    def _valid_indices(self,features):
#        s = np.sum([(features == q) for q in self._filter],0)
#        return s == 0 
#        
#    
#    def _search(self,frames,nresults):
#        start = time()
#        feature = frames[self.feature][::self.step]
#        valid = self._valid_indices(feature)
#        some_valid = np.any(valid)
#        # If any of the frames are valid, remove invalid frames, otherwise,
#        # we have no choice but to use everything.
#        feature = feature[valid] if some_valid else feature
#        
#        if self._fine_feature:
#            ff = frames[self._fine_feature][::self.step]
#            ff = ff[valid] if some_valid else ff
#            ff = self._packer(ff)
#        
#        lf = len(feature)
#        # KLUDGE: Results may not be unique
#        indices = []
#        distances = []
#        for i in range(lf):
#            dist = hamming_distance(feature[i],self._index)
#            srt = np.argsort(dist)
#            
#            # TODO: Make this size configurable
#            # TODO: What effect does altering this size have?
#            n = nresults * 10
#            
#            if self._fine_feature:
#                finer = packed_hamming_distance(ff[i],self._fine_index[srt[:n]])
#                fsrt = np.argsort(finer)
#                srt = srt[:n][fsrt]
#                dist = dist[srt]
#                indices.extend(srt)
#                distances.extend(finer[fsrt])
#            else:
#                indices.extend(srt[:n])
#                distances.extend(dist[srt[:n]])
#        
#        
#        dsrt = np.argsort(distances)
#        indices = np.array(indices)[dsrt[:nresults]]
#        results = [addr for addr in self._addrs[indices]]
#        
#            
#        stop = time() - start
#        print 'search took %1.4f seconds' % stop
#        return results
        
        
    
    
        
    

class LshSearch(FrameSearch):
    # TODO: Replace this with the TypeCodes class in nputil
    # TODO: Implement the _add_index() method
    # TODO: Implement the _check_index() method
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
    def __init__(self,_id,feature,step = None,nbits = None):
        k = LshSearch._DTYPE_MAPPING.keys()
        if nbits not in k:
            raise ValueError('nbits must be in %s') % (str(k))
        
        FrameSearch.__init__(self,_id,feature)
        self._index = None
        self._sorted = None
        self.step = step
        self.nbits = nbits
        
        self._idkey = 'i'
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
    
    def _add_index(self,_id):
        raise NotImplemented()

    def _check_index(self):
        raise NotImplemented()
    
    def _build_index(self):
        '''
        '''
        env = self.env()
        fc = env.framecontroller
        l = int(len(fc) / self.step)
        nids = len(fc.list_ids())
        index = np.recarray(\
                l + nids,dtype = [(self._idkey,np.object),
                           (self._addresskey,np.object),
                           (self._hashkey,self._hashdtype,(self.nbits))])
        

        for i,f in enumerate(fc.iter_all(step = self.step)):
            address,frame = f
            index[i][self._idkey] = frame['_id'][0]
            index[i][self._addresskey] = address
            feature = frame[self.feature]
            try:
                index[i][self._hashkey][:] = self._bit_permute(feature[0])
            except IndexError:
                index[i][self._hashkey][:] = self._bit_permute(feature)
            print index[i][self._hashkey]
        
        index = index[:i]
        # do an argsort for each permutation
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
        # Use self.argsort to get a sorted version of the original block x perm
        # array
        print 'setting up'
        self._sorted = np.ndarray(\
                    self.index[self._hashkey].shape,dtype = self._hashdtype)
        for i in range(self.nbits):
            self._sorted[:,i] = self.index[self._hashkey][:,i][self.argsort[:,i]]
        print 'done setting up'
        
    
    # BUG: This does not return sequences of equal length to the query!!
    def _search(self,frames,nresults):
        if None is self._sorted:
            self._setup()
        
        start = time()
        feature = frames[self.feature][::self.step]
        # Get rid of zeros. This will confuse the results
        nz = np.nonzero(feature)
        feature = feature[nz[0]]
        
        
        lf = len(feature)
        perms = np.zeros((lf,self.nbits))
        
        # Get the permutations of the hash code for every block
        for i,f in enumerate(feature):
            perms[i] = self._bit_permute(f)
        
        # blocks will be rng*2+1 in size
        rng = 10
        
        l = [[] for i in range(lf)]
        # TODO: Parallelize the search
        for i in range(self.nbits):
            # get the insertion positions for every block, for this permutation
            inserts = np.searchsorted(self.sorted[:,i],perms[:,i])
            # get the starting index for every block, for this permutation
            starts = inserts - rng
            starts[starts < 0] = 0
            # get the stopping index for every block, for this permutation
            stops = inserts + rng 
            
            #[l[q].extend(self.argsort[:,i][starts[q] : stops[q]]) for q in range(lf)]
            for q in xrange(lf): 
                l[q].extend(self.argsort[:,i][starts[q] : stops[q]])
        
        
        results = set()
        for candidates in l:
            results.update(Score(candidates).nbest(nresults))
        
        #results = [(r[self._idkey],r[self._addresskey]) for r in results]
        finalresults = []
        for r in results:
            row = self.index[r]
            finalresults.append((row[self._idkey],row[self._addresskey]))
        
        stop = time() - start
        print '_search took %1.4f seconds' % stop
        return finalresults[:nresults]
        
         
        
            
    
           
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
    def __init__(self,_id,feature,step = 1,size = None):
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
    
    def _add_index(self,_id):
        raise NotImplemented()
    
    def _check_index(self):
        raise NotImplemented()
    
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
        starttime = time()
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
        print '_candidate_sequences took %1.4f' % (time() - starttime)
        return candidates
    
    def _score_sequences(self,feature,candidates):
        starttime = time()
        # a list that will hold four-tuples of (_id,address,score,pos)
        finalscores = [] and not isinstance()
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
        print '_score_sequences took %1.4f' % (time() - starttime)
        return finalscores
    
    def _avoid_overlap(self,nresults,finalscores,querylen):
        starttime = time()
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
        
        print '_avoid_overlap took %1.4f' % (time() - starttime)
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
        

            
        