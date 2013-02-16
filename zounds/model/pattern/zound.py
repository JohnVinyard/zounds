from __future__ import division
from copy import deepcopy
from itertools import izip,repeat
from time import time,sleep
from datetime import datetime
from threading import Thread

import numpy as np

from pattern import Pattern
from event import Event
from transform import RecursiveTransform,IndiscriminateTransform
from zounds.util import tostring
from zounds.analyze.feature.rawaudio import AudioFromIterator,AudioFromMemory 
from zounds.pattern import enqueue,init_nrt,render_pattern_non_realtime
from zounds.environment import Environment


# TODO: It'd probably be better to do a max size in bytes, rather than an 
# expiration time in minutes
# KLUDGE: This class doesn't really belong here
class Buffers(Thread):
    
    def __init__(self,expire_time_minutes = 5):
        Thread.__init__(self)
        self.setDaemon(True)
        self._should_run = True
        self._buffers = dict()
        self._expire_seconds = expire_time_minutes * 60
        self.start()
    
    @property
    def env(self):
        return Environment.instance
    
    def has_key(self,key):
        return self._buffers.has_key(key)
    
    def _update(self,key):
        _,audio = self._buffers[key]
        self._buffers[key] = (time(),audio)
        return audio
    
    def __getitem__(self,key):
        return self._update(key)
    
    def allocate(self,p):
        
        try:
            return self._update(p._id)
        except KeyError:
            pass
        
        t = time()
        
        try:
            # p is a FrameModel-derived instance
            audio = self.env.synth(p.audio)
            # BUG: What if this is only a partial segment of a pattern?
            self._buffers[p._id] = (t,audio)
            return audio
        except AttributeError:
            pass
        
        try:
            # p is a leaf pattern, or a pattern that has been analyzed
            fm = self.env.framemodel
            frames = fm[p.address]
            audio = self.env.synth(frames.audio)
            self._buffers[p._id] = (t,audio)
            return audio
        except AttributeError:
            pass
        
        raise ValueError('p must be a FrameModel-derived instance or a Zound')
    
    def run(self):
        while self._should_run:
            keys = self._buffers.keys()
            tm = time()
            for k in keys:
                t,_ = self._buffers[k]
                if tm - t > self._expire_seconds:
                    print 'expiring %s' % k
                    del self._buffers[k]
            
            # check for expired buffers once a minute
            sleep(1000 * 60)
                
    
    def stop(self):
        self._should_run = False


BUFFERS = Buffers()

class Zound(Pattern):
    
    def __init__(self,source = None,external_id = None,_id = None, address = None,
                 pdata = None,all_ids = None,is_leaf = False,stored = False):
        
        # source is the name of the application or user that created this pattern
        self.source = source or self.env().source
        # _id is the zounds _id of this pattern
        self._id = _id or self.env().newid()
        # external_id is the _id assigned to the pattern by the application or
        # user that created it
        self.external_id = external_id or self._id
        Pattern.__init__(self,self._id,self.source,self.external_id)
        self.address = address
        self.pdata = pdata or dict()
        self._sort_events()
        self.all_ids = set(all_ids or [])
        self._patterns = None
        self._is_leaf = is_leaf
        # If the pattern has not yet been stored, this will be None. Otherwise,
        # it will the time at which the pattern was stored in seconds since the
        # epoch
        self.stored = stored
        # keep track of unstored nested patterns that should be stored when
        # self.store() is called
        self._to_store = set()
        
        # INTERNAL USE ONLY!!!
        self._ancestors = []
        # INTERNAL USE ONLY!!!
        # during transforms, keep track of the id this pattern was transformed
        # from
        self._was = None
    
    def _kwargs(self,**kwargs):
        return {}
    
    @property
    def stored_time(self):
        try:
            return datetime.fromtimestamp(self.stored)
        except TypeError:
            return None  
    
    @classmethod
    def list_ids(cls):
        return cls.controller().list_ids()
    
    @classmethod
    def random(cls):
        '''random
        
        Return a random, stored instance
        '''
        _ids = list(cls.list_ids())
        return cls[_ids[np.random.randint(0,len(_ids))]]
        
    def _copy(self,_id,addr):
        return Zound(source = self.source,
                  external_id = _id,
                  _id = _id,
                  address = addr,
                  pdata = deepcopy(self.pdata),
                  all_ids = self.all_ids.copy(),
                  is_leaf = self.is_leaf,
                  stored = False)
    
    def copy(self):
        '''
        Create an exact duplicate of this pattern with a new id.  copy() should
        always be called before modifying a stored pattern. 
        '''
        _id = self.env().newid()
        addr = None if self.address is None else self.address.copy()
        z = self._copy(_id,addr)
        z._to_store = self._to_store.copy()
        if self._patterns:
            z._patterns = self._patterns.copy()
        
        z._ancestors = list(self._ancestors)
        return z
    
    
    def __and__(self,other):
        '''
        overlay two patterns
        '''
        if not other:
            return self.copy()
        
        rn = self.copy()
        p = other.patterns
        for k,v in other.pdata.iteritems():
            # BUG: What if other contains a pattern that this pattern contains?
            # The events in self will be overwritten by the events in other.
            rn.append(p[k],v)
        return rn
    
    def shift(self,amt,recurse = False):
        '''
        Shift events in time by amt.  By default, only top-level patterns are
        altered.  If recurse is True, the transformation is applied at each
        level of nesting
        '''
        if self.is_leaf:
            raise Exception('cannot call shift() on a leaf pattern')
        
        def s(pattern,events):
            if None is events:
                return pattern,events
            return pattern,[e >> amt for e in events]
        
        t = RecursiveTransform(s) if recurse else IndiscriminateTransform(s)
        return self.transform(t)
    
    def __lshift__(self,amt):
        '''
        Shift all times by n seconds, non-recursively
        '''
        return self.shift(-amt)
    
    def __rshift__(self,amt):
        '''
        Shift all times by n seconds, non-recursively
        '''
        return self.shift(amt)
    
    def dilate(self,amt,recurse = True):
        '''
        Multiply all times by factor n
        '''
        if self.is_leaf:
            raise Exception('cannot call dilate() on a leaf pattern')
        
        def s(pattern,events):
            if None is events:
                return pattern,events
            
            return pattern,[e * amt for e in events]
        
        t = RecursiveTransform(s) if recurse else IndiscriminateTransform(s)
        return self.transform(t)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return tostring(self,_id = self._id,source = self.source,
                        external_id = self.external_id,all_ids = self.all_ids,
                        _is_leaf = self._is_leaf,address = self.address)
    
    def __hash__(self):
        return self._id.__hash__()
    
    def __eq__(self,other):
        if self is other:
            return True
        
        eq = (self._id == other._id) and \
             (self.source == other.source) and \
             (self.external_id == other.external_id) and \
             (self.all_ids == other.all_ids) and \
             (self._is_leaf == other._is_leaf)
        
        if self.is_leaf:
            return eq and (self.address == other.address)
        
        return eq
             
    
    @classmethod
    def leaf(cls,addr,source = None):
        '''leaf
        
        :param addr: Address can be a Zounds frame id, a Zounds frame address, or \
        a Frames-derived instance with the address property set
        '''
        e = cls.env()
        source = source or e.source
        
        if isinstance(addr,e.address_class):
            # addr is a backend-specific address
            return Zound(source = source,address = addr,is_leaf = True)
        
        
        try:
            # address is a Frames id
            a = e.framecontroller.address(addr)
            return Zound(source = source,address = a,is_leaf = True) 
        except KeyError:
            pass
        
        try:
            if addr.address:
                # addr is a frames instance
                if isinstance(addr.address,str):
                    a = e.framecontroller.address(addr.address)
                else:
                    a = addr.address
                return Zound(\
                        source = source,address = a,is_leaf = True)
        except AttributeError:
            pass
        
        raise ValueError('addr must be a Zounds address, a Zounds frame _id, or a Frames instance with the address property set')
            
        
        
    @property
    def is_leaf(self):
        return self._is_leaf 
    
    
    # TODO: Move this into the base Pattern class
    # BUG: What if a transform changes the length of the samples?
    def length_samples(self,**kwargs):
        '''
        The length of this pattern in samples, when rendered as raw audio
        '''
        try:
            # this pattern has been analyzed and is in the frames database,
            # so it's trivial to find out its length in samples
            
            # BUG: This won't work for stored MusicPattern instances if
            # their bpm value has changed!
            return self.env().frames_to_samples(len(self.address))
        except TypeError:
            sr = self.env().samplerate
            # this pattern hasn't yet been analyzed, so we have to calculate
            # its length in samples
            last = 0
            patterns = self.patterns
            for k,v in self.pdata.iteritems():
                pattern = patterns[k]
                # get the latest end time of all the events
                total = \
                    max([(self.interpret_time(e.time,**kwargs) * sr) + e.length_samples(pattern,**kwargs) \
                          for e in v])
                if total > last:
                    last = total
            
            return last
    
    # TODO: Move this into the base Pattern class
    def length_seconds(self,**kwargs):
        '''
        The length of this pattern in seconds, when rendered as raw audio
        '''
        return self.length_samples(**kwargs) / self.env().samplerate
    
    
    def audio_extractor(self,needs = None):
        e = self.env()
        
        if self.address:
            # this pattern has already been rendered, so we can just grab 
            # encoded audio from the database and render it
            return AudioFromMemory(e.samplerate,e.windowsize,e.stepsize,
                                   self._render(),needs = needs)
        
        # this pattern has not yet been rendered and stored
        return AudioFromIterator(e.samplerate,e.windowsize,e.stepsize,
                                 self.iter_audio(),needs = needs)
    
    def iter_audio(self,**kwargs):
        # ensure that everything is setup
        init_nrt()
        # enqueue this pattern in the non-realtime queue
        enqueue(self,BUFFERS,self.env().samplerate,realtime = False)
        env = self.env()
        # allocate memory for the audio buffer
        buf = np.zeros(env.chunksize,dtype = np.float32)
        
        rpnr = render_pattern_non_realtime
        
        # start rendering chunks
        time = 0
        frames_filled = rpnr(buf.size,time,buf)
        
        while frames_filled == buf.size:
            yield buf
            buf[:] = 0
            time += frames_filled
            frames_filled = rpnr(buf.size,time,buf)
        
        if frames_filled:
            yield buf[:frames_filled]
        
        # cleanup
        init_nrt()
            
    
    
    def _render(self,**kwargs):
        # render the pattern as audio
        
        # KLUDGE: Maybe _render should be an iterator, for very long patterns
        
        if self.empty:
            raise Exception('Cannot render an empty pattern')
        
        env = self.env()
        if self.address:
            # this is a "leaf" pattern that has already been rendered and analyzed,
            # so it can just be retrieved from the data store
            return env.synth(env.framemodel[self.address].audio)
    
        
        # KLUDGE: What about very long patterns?
        # make a *guess* about the amount of memory needed to hold the entire
        # pattern.  Note that some transforms may make the pattern longer
        # than we expect. 
        fudge = 10 * env.samplerate
        audio = np.zeros(\
                self.length_samples(**kwargs) + fudge,dtype = np.float32)
        
        time = 0
        for chunk in self.iter_audio(**kwargs):
            stop = time + chunk.size
            
            if stop >= audio.size:
                print 're-allocating'
                # our guess, even with the fudge-factor, was too small.
                # Re-allocate memory.
                extra = np.zeros(stop - audio.size,dtype = np.float32)
                audio = np.concatenate([audio,extra])
            
            audio[time : stop] = chunk
            
            time += chunk.size
        
        return audio[:time]
    
    def interpret_time(self,time,**kwargs):
        '''
        Patterns might interpret time values in different ways.  The default
        interpretation is the identity function.  Times are expressed in seconds.
        '''
        return time
    
    # TODO: Test
    # TODO: Since patterns are immutable, is it safe to store the result of
    # this once it has been run once?  I *think* so.
    def _leaves_absolute(self,d = None,patterns = None,offset = 0,**kwargs):
        '''
        Get a dictionary mapping leaf patterns to events with *absolute* times
        '''
        if self.is_leaf:
            return {self._id : [Event(offset)]}
        
        if None is d:
            d = dict()
        
        if not patterns:
            patterns = self.patterns
        
        # BLEGH!! This is ugly!
        
        # iterate over each child pattern
        for k,v in self.pdata.iteritems():
            p = patterns[k]
            # iterate over the events for this pattern
            for e in v:
                time = self.interpret_time(e.time,**kwargs)
                l = p._leaves_absolute(\
                        d = d, patterns = patterns,offset = offset + time,**kwargs)
                
                if not p.is_leaf:
                    continue
                
                for _id,events in l.iteritems():
                    try:
                        d[_id].extend(events)
                    except KeyError:
                        d[_id] = events
            
        return d
    
    
    def play(self,time = 0):
        '''
        play this pattern in realtime, starting time seconds from now. This method
        assumes that the realtime audio server is running.
        '''
        enqueue(self,BUFFERS,self.env().samplerate)
    
    def _sort_events(self):
        for v in self.pdata.itervalues():
            v.sort()
    
    # TODO: Tests
    @property
    def patterns(self):
        
        if None is self._patterns:
            
            # fetch all the ids that are stored at once
            plist = self.__class__\
                [self.all_ids - set((p._id for p in self._to_store))]
            
            # add unstored patterns
            plist.extend(self._to_store)
            # create a dictionary mapping pattern id -> pattern
            self._patterns = dict((p._id,p) for p in plist)
            return self._patterns
        
        
        for _id in self.all_ids:
            if not self._patterns.has_key(_id):
                p = self.__class__[_id]
                self._patterns[p._id] = p
        
        
        return self._patterns
    
    
    def _check_stored(self):
        if self.stored:
            raise Exception('Cannot modify a stored pattern')
    
    @property
    def empty(self):
        
        if self.is_leaf:
            return False
        
        return not bool(sum([len(v) for v in self.pdata.itervalues()]))
    
    def _empty_pattern(self,was = None):
        p = Zound(source = self.source)
        p._was = was
        return p
    
    
    def append(self,pattern,events):
        '''append
        
        Add a pattern at one or more locations in time to this pattern
        
        :param _id: the _id of the pattern to be added
        
        :param events: a list of two-tuples of (time_secs,transformations)
        '''
        
        # raise an exception if this pattern has already been stored
        self._check_stored()
        
        if pattern._id == self._id:
            raise ValueError('Patterns cannot contain themselves!')
        
        if not events:
            raise ValueError('events was empty')
        
        try:
            l = self.pdata[pattern._id]
        except KeyError:
            l = []
            self.pdata[pattern._id] = l
        
        # add events and sort them in ascending distance from "now"
        l.extend(events)
        l.sort()
        
        # update the flat list of all ids required to render this pattern
        self.all_ids.add(pattern._id)
        self.all_ids.update(pattern.all_ids)
        
        # update the list of nested patterns which have not yet been stored
        if not pattern.stored:
            self._to_store.add(pattern)
            if self._patterns is None:
                self._patterns = dict()
            
            self._patterns[pattern._id] = pattern
            self._patterns.update(pattern.patterns)
    
    
    def _leaf_compare(self,other):
        if not self.is_leaf:
            raise Exception('%s is not a leaf' % self)
        
        return self.address == other.address
    
    def _interpret_transform_result(self,p_orig,pt,et,pattern_queue):
        '''
        :param p_orig: the pattern that was passed to the transform
        
        :param e_orig: the events list that was passed to the transform
        
        :param pt: the pattern or patterns that were returned by the transform
        
        :param et: the events list(s) that were returned by the transform
        
        :param pattern_queue: a list of patterns that are yet-to-be transformed
        '''
        if isinstance(pt,Zound):
            if pt == p_orig:
                # the original pattern was returned by the transform
                return p_orig,et
            
            # a single pattern that wasn't the original was returned by the
            # transform
            pattern_queue.append((pt,et))
            
            # return None events for the original pattern so it will be removed
            return p_orig,None
        
        # pt wasn't a Zound.  Assume that pt is a list.
        po = p_orig
        eo = []
        
        try:
            index = pt.index(p_orig)
            po = pt.pop(index)
            eo = et.pop(index)
        except ValueError:
            pass
        
        # append any new patterns and events to the transform queue
        for i in xrange(len(pt)):
            pattern_queue.append((pt[i],et[i]))
        
        return po,eo
    
    def transform(self,transform,changed = None,top = True):
        
        # TODO: Ensure that transform isn't None, and has at least one
        # transformation defined
        
        
        if self.is_leaf:
            n = self.__class__(source = self.source,
                               address = self.address,
                               is_leaf = self.is_leaf)
            t = transform(n, changed = changed, top = top)
            # this is a leaf pattern, and it wasn't altered in any way, so
            # return self. Otherwise, return the modified pattern
            return self if self._leaf_compare(t) else t
        
        # create a new, empty pattern
        n = self._empty_pattern(was = self)
        
        # create a queue of patterns that need to be transformed
        pq = list(self.iter_patterns())
        
        changed = [False]
        
        while pq:
            
            pattern,events = pq.pop()
            p,e = pattern,events
            
            p._ancestors.append(self)
            p._ancestors.extend(self._ancestors)
            
            try:
                p,e = transform(pattern,events,changed = changed, top = False)
                # Interpret the result of the transform
                p,e = self._interpret_transform_result(pattern, p, e, pq)
                if not e:
                    # No events were returned, so the pattern is being removed
                    continue
                # Call _follow_up, if necessary
                p,e = transform._follow_up(p,e,changed = changed,top = False)
            except KeyError as ke:
                if not top:
                    raise ke
                
                # there was no transform defined for this pattern
                pass
            
            p._ancestors = []
            n.extend(p,e)
        
        return n
            
    
    
    def extend(self,patterns,events):
        '''
        single pattern, multiple events -> pattern,[]
        multi pattern, single event each -> [],[]
        multi pattern, multi event each -> [],[]
        '''
        
        try:
            l = patterns.__len__()
        except AttributeError:
            # patterns is single pattern instance
            if events:
                self.append(patterns,events)
            return
        
        if isinstance(events,Event):
            # events is a single Event instance. Create an infinite generator,
            # so this event will be added once for every pattern
            events = repeat(events)  
        elif l != len(events):
            raise ValueError(\
            'If patterns is an iterable, patterns and events must have the same length')
        
        
        for args in izip(patterns,events):
            if not args[1]:
                # skip patterns with no events
                continue
            
            self.append(*args)
    
    # TODO: Tests
    # TODO: Is this method necessary, or is it made superfluous by transform()
    def remove(self,pattern_id = None, criteria = None):
        self._check_stored()
        # TODO: be sure to remove items from all_ids and _to_store, when necessary
        raise NotImplemented()
    
    
    def iter_patterns(self):
        p = self.patterns
        for k,e in self.pdata.iteritems():
            yield p[k],e
    
    def __iter__(self):
        yield self
    
    
    # TODO: Tests
    def __getitem__(self,key):
        '''
        get specific patterns, or time slices
        '''
        
        raise NotImplemented()
        
        exc_msg = 'key must be a string or a slice'
        if not key:
            raise ValueError(exc_msg)
        
        try:
            # treat key as a pattern key
            return self.patterns[key]
        except KeyError:
            pass
        
        try:
            # treat key as a slice of time with start and stop seconds defined
            # start and stop must both be positive
            # TODO: What happens if the slice bisects a pattern?  Should events
            # be flattened first, or does preserving the hierachical structure
            # make sense?
            pass
        except AttributeError:
            raise ValueError(exc_msg)
    
    
    def todict(self):
        d = {
             '_id' : self._id,
             'source' : self.source,
             'external_id' : self.external_id,
             'is_leaf' : self._is_leaf,
             'stored' : self.stored,
             '_type'  : self.__class__.__name__
             }
        
        if self.address:
            d['address'] = self.address.todict()
        
        d['all_ids'] = list(self.all_ids)
        
        pdata = dict()
        for k,v in self.pdata.iteritems():
            pdata[k] = [e.todict() for e in v]
        d['pdata'] = pdata
        return d
    
    def todict_comprehensive(self):
        '''
        Return a dictionary containing this pattern and all patterns
        required to render it.
        '''
        d = dict()
        d['root'] = self.todict()
        
        patterns = self.__class__[self.all_ids]
        d['patterns'] = dict([(p._id,p.todict()) for p in patterns])
        d['_type'] = self.__class__.__name__
        return d 
    
    @classmethod
    def fromdict(cls,d,stored = False):
        
        # KLUDGE: Given the following implementation, this copy() is necessary for
        # the InMemory controller implementation, but is probably unnecessary
        # and inefficient for "real" implementations 
        d = d.copy()
        
        try:
            address = d['address']
            addr_cls = cls.env().address_class
            d['address'] = addr_cls.fromdict(address) if address else None
        except KeyError:
            pass
        
        if d['is_leaf']:
            d['all_ids'] = None
            d['pdata'] = None
        else:
            pdata = {}
            for k,v in d['pdata'].iteritems():
                pdata[k] = [Event.fromdict(e) for e in v]
            d['pdata'] = pdata
        
        # KLUDGE: Why is this here?  When would you pass stored = False to this
        # method?
        if not stored:
            d['stored'] = stored
        
        _type = d['_type']
        del d['_type']
        return cls._impl[_type](**d)
    
    # TODO: Should this be asynchronous ?
    def store(self):
        
        if self.empty:
            # KLUDGE: Should I define a custom exception for this?
            raise Exception('Cannot store an empty pattern')
        
        # store any nested unstored patterns 
        for p in self._to_store:
            p.store()
        
        # store self
        self.__class__._store(self)
