from __future__ import division
from copy import deepcopy
from abc import ABCMeta,abstractmethod
from itertools import izip,repeat
from threading import Thread
from time import sleep,time

import numpy as np

from zounds.model.model import Model
from zounds.analyze.feature.rawaudio import AudioFromDisk,AudioFromMemory
from zounds.analyze.synthesize import TransformChain
from zounds.util import tostring
from zounds.environment import Environment
from zounds.pattern import usecs,put

class MetaPattern(type):
    
    def __init__(self,name,bases,attrs):
        super(MetaPattern,self).__init__(name,bases,attrs)
    
    def __getitem__(self,key):
        item = self.controller()[key]
        
        try:
            return self.fromdict(item,stored = True)
        except AttributeError:
            return [self.fromdict(i,stored = True) for i in item]
    
    def _store(self,pattern):
        self.controller().store(pattern.todict())
        pattern.stored = True

class Pattern(Model):
    '''
    A Pattern is the central concept in zounds.  Patterns can be nested.
    a "leaf" pattern represents a list of ids which point to audio frames.
    A "branch" pattern points to other patterns.
    '''
    __metaclass__ = MetaPattern
    
    def __init__(self,_id,source,external_id):
        Model.__init__(self)
        
        self.source = source
        self.external_id = external_id
        self._id = _id
        self._data = {'source'      : self.source,
                      'external_id' : self.external_id,
                      '_id'         : self._id} 
        
        self._fc = None
    
    @property
    def framecontroller(self):
        if not self._fc:
            self._fc = self.env().framecontroller
        return self._fc
    
    
    def audio_extractor(self,needs = None):
        raise NotImplemented()
    
    
    def data(self):
        return self._data

class FilePattern(Pattern):
    '''
    Represents a pattern in the form of an audio file on disk that has not 
    yet been stored 
    '''
    
    def __init__(self,_id,source,external_id,filename):
        Pattern.__init__(self,_id,source,external_id)
        self.filename = filename

    def audio_extractor(self, needs = None):
        e = self.__class__.env()
        return AudioFromDisk(e.samplerate,
                             e.windowsize,
                             e.stepsize,
                             self.filename,
                             needs = needs)

class DataPattern(Pattern):
    '''
    Represents a pattern in the form of an in-memory numpy array of audio 
    samples that has not yet been stored
    '''
    def __init__(self,_id,source,external_id,samples):
        Pattern.__init__(self,_id,source,external_id)
        self.samples = samples
        
    def audio_extractor(self,needs = None):
        e = self.__class__.env()
        return AudioFromMemory(e.samplerate,
                               e.windowsize,
                               e.stepsize,
                               self.samples,
                               needs = needs)

# KLUDGE: I'm not sure where this class belongs
# TODO: It'd probably be better to do a max size in bytes, rather than an 
# expiration time in minutes

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
            self._update(p._id)
            return
        except KeyError:
            pass
        
        t = time()
        
        try:
            # p is a FrameModel-derived instance
            audio = self.env.synth(p.audio)
            self._buffers[p._id] = (t,audio)
            return
        except AttributeError as e:
            print e
            pass
        
        try:
            # p is a leaf pattern, or a pattern that has been analyzed
            fm = self.env.framemodel
            frames = fm[p.address]
            audio = self.env.synth(frames.audio)
            self._buffers[p._id] = (t,audio)
            return
        except AttributeError as e:
            print e
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
    
class Event(object):
    
    def __init__(self,time,*args):
        '''__init__
        
        :param time_secs: The time in seconds at which the event should occur
        
        :param kwargs: A dictionary mapping zounds.analyze.synthesize.Transform \
        derived class names to the parameters for that transform
        '''
        object.__init__(self)
        self.time = time
        self.transforms = args
    
    def __copy__(self):
        return Event(self.time,*deepcopy(self.transforms))
    
    def __eq__(self,other):
        return self.time == other.time
    
    def __ne__(self,other):
        return self.time != other.time
    
    def __lt__(self,other):
        return self.time < other.time
    
    def __lte__(self,other):
        return self.time <= other.time
    
    def __gt__(self,other):
        return self.time > other.time
    
    def __gte__(self,other):
        return self.time >= other.time
    
    def shift(self,amt):
        return Event(self.time + amt,*deepcopy(self.transforms))
        
    def __lshift__(self,amt):
        return self.shift(-amt)
    
    def __rshift__(self,amt):
        return self.shift(amt)
    
    def __add__(self,amt):
        return self.shift(amt)
    
    def __radd__(self,amt):
        return Event(amt + self.time,*deepcopy(self.transforms))
    
    def __sub__(self,amt):
        return self.shift(-amt)
    
    def __rsub__(self,amt):
        return Event(amt - self.time,*deepcopy(self.transforms))
    
    def __mul__(self,amt):
        return Event(self.time * amt,*deepcopy(self.transforms))
    
    def __neg__(self):
        '''
        Negate the time at which this event occurs.  Note that the interpretation
        of a negative time value is up to the containing pattern, and may not
        always be valid.
        '''
        return Event(-self.time,*deepcopy(self.transforms))
    
    def __iter__(self):
        yield self
    
    def todict(self):
        return Event.encode_custom(self)
    
    @classmethod
    def fromdict(cls,d):
        return Event.decode_custom(d) 
    
    @staticmethod
    def encode_custom(event):
        # TODO: This should include transform data too
        return {'time' : event.time}
    
    @staticmethod
    def decode_custom(doc):
        # TODO: This should include transform data too
        return Event(doc['time'])
    
    def length_samples(self,pattern,**kwargs):
        # TODO: This should check the pattern against all transforms to decide
        # if the transform might shorten or lengthen the pattern
        
        # TODO: Are there situations where the pattern must be rendered with the
        # transform to calculate its new length?
        return pattern.length_samples(**kwargs)

    def __str__(self):
        return tostring(self,time = self.time)
    
    def __repr__(self):
        return self.__str__()


# TODO: Composable types with different atomic behaviors
class BaseTransform(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def _get_transform(self,pattern,events = None):
        '''
        Try to get a transform for the pattern and events.  If one doesn't
        exist, throw a KeyError
        '''
        raise NotImplemented()
    
    def __call__(self,pattern,events = None, changed = False, top = True):
            
        # check if there's a transform defined for this pattern
        t = self._get_transform(pattern, events)
        
        if isinstance(t,BaseTransform):
            return [t(pattern)],[events]
        
        # t will return a two-tuple of pattern,events
        return t(pattern,events)
    
    def _follow_up(self,pattern,events = None,changed = False,top = True):
        return pattern,events

class RecursiveTransform(BaseTransform):
    
    def __init__(self,transform,predicate = None):
        BaseTransform.__init__(self)
        self.transform = transform
        self.predicate = predicate or (lambda p,e: True)
    
    def _get_transform(self,pattern,events = None):
        return self.transform

    def __call__(self,pattern,events = None,changed = None, top = True):
        
        if self.predicate(pattern,events):
            # the predicate matched this pattern. transform it.
            p,e = self.transform(pattern,events)
            changed[0] = True
        else:
            # the predicate didn't match. leave the pattern and its events
            # unaltered
            p,e = pattern,events
        
        if events is None:
            # this is a leaf pattern
            if not changed[0]:
                # no patterns in this branch have changed
                raise KeyError
            return p
        
        return p,e
    
    def _follow_up(self,pattern,events = None,changed = None,top = True):
            
        return pattern.transform(self,changed = changed, top = top), events

class IndiscriminateTransform(BaseTransform):
    
    def __init__(self,transform):
        BaseTransform.__init__(self)
        self.transform = transform
    
    def _get_transform(self,pattern,events = None):
        return self.transform

class ExplicitTransform(BaseTransform,dict):
    
    def __init__(self,transforms):
        BaseTransform.__init__(self)
        dict.__init__(self,transforms)
    
    def _get_transform(self,pattern,events = None):
        return self[pattern._id]

class CriterionTransform(BaseTransform):
    
    def __init__(self,predicate,transform):
        BaseTransform.__init__(self)
        self.predicate = predicate
        self.transform = transform
    
    def _get_transform(self,pattern,events = None):
        if self.predicate(pattern,events):
            return self.transform
        
        raise KeyError()        
        
        
# TODO: Add created date
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
                print v
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
    
    
    def _render(self,**kwargs):
        # render the pattern as audio
        
        # KLUDGE: Maybe _render should be an iterator, for very long patterns
        
        # TODO: rendering should happen *just like* realtime playing, i.e.,
        # audio rendering should be handled by the audio back-end in 
        # "free-wheeling" mode.
        
        if self.empty:
            raise Exception('Cannot render an empty pattern')
        
        env = self.env()
        if self._is_leaf:
            # this is a "leaf" pattern that has already been rendered and analyzed,
            # so it can just be retrieved from the data store
            return env.synth(env.framemodel[self.address].audio)
    
        # allocate memory to hold the entire pattern
        audio = np.zeros(self.length_samples(**kwargs),dtype = np.float32)
        
        patterns = self.patterns
        for k,v in self.pdata.iteritems():
            # render the sub-pattern
            p = patterns[k]
            a = p._render()
            for event in v:
                # render each occurrence of the sub-pattern
                # TODO: Don't perform the same transformation twice!
                time,tc = self.interpret_time(event.time,**kwargs), \
                          TransformChain.fromdict(event.transforms)
                ts = int(time * env.samplerate)
                transformed = tc(a)
                # apply any transformations and add the result to the output
                audio[ts : ts + len(transformed)] += transformed
            
        return audio
    
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
        play this pattern in realtime, starting time seconds from now
        '''
        # get all leaf patterns with *Absolute* times
        leaves = self._leaves_absolute()
        patterns = self.patterns
        # allocate buffers for them
        if self.is_leaf:
            BUFFERS.allocate(self)
        else:
            for k in leaves.iterkeys():
                BUFFERS.allocate(patterns[k])
        
        # schedule them
        now = usecs()
        # TODO: Stress test/profile the play method and find out what an 
        # acceptable latency value is. I think this value is too high.
        latency = .25 * 1e6
        for k,v in leaves.iteritems():
            audio = BUFFERS[k]
            la = len(audio)
            for e in v:
                put(audio,0,la,now + latency + (e.time * 1e6))
        
    
    def audio_extractor(self,needs = None):
        e = self.env()
        return AudioFromMemory(e.samplerate,
                               e.windowsize,
                               e.stepsize,
                               self._render(),
                               needs = needs)
    
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
             'stored' : self.stored
             }
        
        if self.address:
            d['address'] = self.address.todict()
        
        d['all_ids'] = list(self.all_ids)
        
        pdata = dict()
        for k,v in self.pdata.iteritems():
            pdata[k] = [e.todict() for e in v]
        d['pdata'] = pdata
        return d
    
    @classmethod
    def fromdict(cls,d,stored = False):
        
        # KLUDGE: Given the following implementation, this copy() is necessary for
        # the InMemory controller implementation, but is probably unnecessary
        # and inefficient for "real" implementations 
        d = d.copy()
        
        if d.has_key('address'):
            d['address'] = cls.env().address_class.fromdict(d['address'])
        
        if d['is_leaf']:
            # BUG: The presence of an addreess is being used to decide if the
            # pattern is a leaf pattern, but all patterns will be analyzed and
            # given an address.  This is broken for non-leaf patterns that have
            # been analyzed.
            d['all_ids'] = None
            d['pdata'] = None
        else:
            pdata = {}
            for k,v in d['pdata'].iteritems():
                pdata[k] = [Event.fromdict(e) for e in v]
            d['pdata'] = pdata
        
        d['stored'] = stored
        return cls(**d)
    
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


class MusicPattern(Zound):
    '''
    Things to think about:
    
    - Q: Does changing a pattern's tempo mean that it should be copied and re-analyzed?
      This seems like it might lead to lots of nearly-identical copies of patterns
      A: No. Changing tempo doesn't cause a copy to be made.
    
    - Q: Should "wrap" be an instance-level attribute?
      A: Wrap will be the default, for now.  This means that there's an underlying
      assumption that patterns are loops, which I'm not sure I like, totally, 
      but it's ok for now.
    
    - Q: Should leaf patterns be instances of Zound, or MusicPattern.  bpm might
      make sense for some leaf patterns (a drum loop), but not for others
      (strumming a guitar chord and letting it sustain)
      A: For now, all leaf patterns will be Zound instances
    
    - What else?
    '''
    
    def __init__(self,source = None,external_id = None, _id = None,address = None,
                 pdata = None,all_ids = None,is_leaf = False,stored = False,
                 bpm = 120,length_beats = 4):
        
        Zound.__init__(self,source = source,external_id = external_id,
                       address = address, pdata = pdata, all_ids = all_ids, 
                       is_leaf = is_leaf, stored = stored,_id = _id)
        self.bpm = bpm
        self.length_beats = length_beats
    
    def _kwargs(self,**kwargs):
        bpm = 'bpm'
        return {bpm : kwargs[bpm] if bpm in kwargs else self.bpm}
    
    def _render(self,**kwargs):
        return Zound._render(self,**self._kwargs(**kwargs))
    
    def length_samples(self,**kwargs):
        return Zound.length_samples(self,**self._kwargs(**kwargs))
    
    def length_seconds(self,**kwargs):
        return Zound.length_seconds(self,**self._kwargs(**kwargs))
    
    def _copy(self,_id,addr):
        return MusicPattern(source = self.source,
                  external_id = _id,
                  _id = _id,
                  address = addr,
                  pdata = deepcopy(self.pdata),
                  all_ids = self.all_ids.copy(),
                  is_leaf = self.is_leaf,
                  stored = False,
                  length_beats = self.length_beats,
                  bpm = self.bpm)
    
    def __and__(self,other):
        mp = Zound.__and__(self,other)
        mp.length_beats = max(self.length_beats,other.length_beats)
        mp.bpm = self.bpm
        return mp
    
    def __add__(self,other):
        '''
        Concatenate two patterns so that other occurs immediately after self.  If
        the two patterns have different tempos, the tempo of the left-hand side
        is kept.
        '''
        if not other:
            return self.copy()
        
        if self.is_leaf or other.is_leaf:
            raise ValueError('Cannot add leaf patterns')
        
        # TODO: Maybe copy should rectify any negative or wrapped times 
        rn = self.copy()
        p = other.patterns
        for k,v in other.pdata.iteritems():
            # BUG: What if other contains a pattern that this pattern contains?
            # The events in self will be overwritten by the events in other.
            
            # BUG: What if self has negative or wrapped event times? The same
            # problems apply.
            
            pattern = p[k]
            events = []
            for e in v:
                time = pattern._interpret_beats(e.time) + rn.length_beats
                events.append(Event(time))
            rn.append(pattern,events)
        rn.length_beats += other.length_beats
        return rn
    
    def __radd__(self,other):
        '''
        Implemented so that an iterable of patterns can be sum()-ed
        '''
        if not other:
            return self.copy()
        
        return self + other
    
    def __mul__(self,n):
        '''
        Repeat this pattern n times 
        '''
        if self.is_leaf:
            raise Exception('cannot multiply a leaf pattern')
        
        container = MusicPattern(source = self.source,
                                 bpm = self.bpm,
                                 length_beats = self.length_beats * n)
        container.append(self,[Event(i * self.length_beats) for i in xrange(n)])
        return container
    
    def __invert__(self):
        '''
        Reverse a pattern with the ~ operator.
        '''
        if self.is_leaf:
            raise Exception('cannot invert a leaf pattern')
        
        def s(pattern,events):
            if None is events:
                return pattern,events
            
            ev = [-(e + 1) for e in events]
            
            return pattern,ev
         
        return self.transform(RecursiveTransform(s))
    
    def _interpret_beats(self,time):
        '''
        Interpret any negative times or wrapped beats
        '''
        actual_beats = time % self.length_beats
        if actual_beats < 0:
            actual_beats = self.length_beats - actual_beats
        return actual_beats
    
    def interpret_time(self,time,**kwargs):
        actual_beats = self._interpret_beats(time)
        
        if kwargs:
            bpm = kwargs['bpm']
        else:
            bpm = self.bpm
        return actual_beats * (1 / (bpm / 60))
    
    def _leaves_absolute(self,d = None,patterns = None,offset = 0,**kwargs):
        return Zound._leaves_absolute(self, d = d, patterns = patterns, 
                                      offset = offset, **self._kwargs(**kwargs))
    
    def _empty_pattern(self,was = None):
        mp = MusicPattern(source = self.source,
                            length_beats = self.length_beats,
                            bpm = self.bpm)
        mp._was = was
        return mp 
    
    def todict(self):
        '''
        Include bpm and length_beats data
        '''
        d = Zound.todict(self)
        d['length_beats'] = self.length_beats
        d['bpm'] = self.bpm
        return d    