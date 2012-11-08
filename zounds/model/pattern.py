'''
Ann - The user
Client - A client application. Maybe a html+javascript app, a user in the python
repl, or a native GUI application.

Ann finds a portion of sound she'd like to use: one kick drum hit.

Pattern from the outside world, i.e., just some sound file
{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,200) // A PyTables address
}

 >>> p = Pattern.leaf(addr) // addr can be a frames instance, an address, or a frame id
 >>> p.store()

She store()s the pattern.

Ann then arranges the kick pattern into one in which the drum plays four times
in succession. She store()s this pattern.

  >>> p2 = Pattern()
  >>> p2.append(p,[Event(0,amp = 1),Event(1,amp = .5),...])
  >>> p2.store()
  >>> _id = p2._id


{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,200) // A PyTables address
}

A pattern using that slice
{
    _id : 'fourquarter'    
    source : Ann
    external_id : 'fourquarter' 
    address : None
    all_ids : ['leaf'],
    data : {
        'leaf' : [
            (0, [{'amp' : (1,)}]),
            (1, [{'amp' : (.5,)}]),
            (2, [{'amp' : (1,)}]),
            (3, [{'amp' : (.5,)}]),
        ],
        
        'branch' : [
            (4, [{'amp' : (1)}])
        ]
    }
}

 

The client is returned a data structure like the one above.  Notice that address
is None.  'fourquarter' has been placed into a queue to be analyzed, and doesn't
yet have a home in the frames database.  Notice that 'leaf' is also returned,
so the pattern can be rendered by the client.  

Ann closes the client application, and comes back later.  She fetches 'fourquarter' from
the database.  Note that 'fourquarter' now has an address.  This means
it has been analyzed, and is a candidate for audio-similarity results.  Ann 
decides she'd like the beats to play only on the 1 and 3.  Note that the 
'fourquarter' pattern is returned to the client, as well as all patterns it 
references.


** IDEA ** Note that patterns are an example of an audio encoding.  They can
be interpreted by a synthesizer. It would be very space efficient to store the
JSON or BSON data instead of raw audio, in most cases.  

{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,200) // A PyTables address
}

{
    _id : 'fourquarter'    
    source : Ann
    external_id : 'fourquarter' 
    address : (100,500)
    all_ids : ['slice'],
    data : {
        'leaf' : [
            (0, [{'amp' : (1,)}]),
            (2, [{'amp' : (1,)}]),
        ]
    }
}

 >>> p3 = Pattern[_id]
 >>> p4 = p3.remove('leaf',[1,3])
 >>> p4.store()
 >>> _id = p4._id

She calls store() on this pattern.  When the data is sent back to the server, 
leaf's hash value is the same, but 'fourquarter's hash value has changed.  'fourquarter'
is stored as a new pattern 'twoquarter'.

Ann returns to the client later, and fetches the 'twoquarter' pattern.

{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,200) // A PyTables address
}

{
    _id : 'twoquarter'    
    source : Ann
    external_id : 'twoquarter' 
    address : (100,500)
    all_ids : ['leaf'],
    data : {
        'leaf' : [
            (0, [{'amp' : (1,)}]),
            (2, [{'amp' : (1,)}]),
        ]
    }
}

She decides that the second beat should be shorter than the first.  This changes
the 'leaf' pattern from

{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,200) // A PyTables address
}

to 

{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,180) // A PyTables address
}

 >>> p4 = Pattern[_id]
 >>> leaf = p4.get('leaf',2)[0]
 >>> leaf_shorter = leaf.change_address((100,180))
 >>> p4 = p4.replace((leaf,2),leaf_shorter)
 >>> p4.store()

When she alters the already stored leaf pattern, the client makes a copy of the
pattern. When she store()s 'twoquarter', three patterns are sent back to the 
server:

{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf' 
    address : (100,200) // A PyTables address
}

{
    _id : 'leaf'
    source : Ann
    external_id : 'leaf2' 
    address : (100,180) // A PyTables address
}

{
    _id : 'twoquarter'    
    source : Ann
    external_id : 'twoquarter' 
    address : (100,500)
    all_ids : ['leaf','leaf2'],
    data : {
        'leaf' : [
            (0, [{'amp' : (1,)}])
        ],
        'leaf2' : [
            (2, [{'amp' : (1,)}])
        ]
    }
}

The second leaf pattern has a different hash value than it was sent with, and
so a new leaf pattern is created.

All changes to a pattern should happen by calling methods on the Pattern class.
No direct acess to the underlying data structures should be made. Maybe the
last operation Ann executes would happen like so: 
'''
from copy import deepcopy
from abc import ABCMeta,abstractmethod
from itertools import izip,repeat

import numpy as np

from zounds.model.model import Model
from zounds.analyze.feature.rawaudio import AudioFromDisk,AudioFromMemory
from zounds.analyze.synthesize import TransformChain
from zounds.util import tostring

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



class Event(object):
    
    def __init__(self,time_secs,**kwargs):
        '''__init__
        
        :param time_secs: The time in seconds at which the event should occur
        
        :param kwargs: A dictionary mapping zounds.analyze.synthesize.Transform \
        derived class names to the parameters for that transform
        '''
        object.__init__(self)
        self.time = time_secs
        self.params = kwargs
    
    def __copy__(self):
        return Event(self.time,**deepcopy(self.params))
    
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
        return Event(self.time + amt,**deepcopy(self.params))
        
    def __lshift__(self,amt):
        return self.shift(-amt)
    
    def __rshift__(self,amt):
        return self.shift(amt)
    
    def __mul__(self,amt):
        return Event(self.time * amt,**deepcopy(self.params))
    
    def __iter__(self):
        yield self
    
    def todict(self):
        raise NotImplemented()
    
    def fromdict(self):
        raise NotImplemented()


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

class RecursiveTransform(BaseTransform):
    
    def __init__(self,transform,predicate = None):
        BaseTransform.__init__(self)
        self.transform = transform
        self.predicate = predicate or (lambda p,e: True)
    
    def _get_transform(self,pattern,events = None):
        return self.transform
    
    def __call__(self,pattern,events = None,changed = False, top = True):
        
        if self.predicate(pattern,events):
            p,e = self.transform(pattern,events)
            changed = True
        else:
            p,e = pattern,events
        
        if events is None:
            # this is a leaf pattern
            if not changed:
                # no patterns in this branch have changed
                raise KeyError
            return p
        
        return p.transform(self, changed = changed, top = top),e

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
                 data = None,all_ids = None,is_leaf = False,stored = False):
        
        # source is the name of the application or user that created this pattern
        self.source = source or self.env().source
        # _id is the zounds _id of this pattern
        self._id = _id or self.env().newid()
        # external_id is the _id assigned to the pattern by the application or
        # user that created it
        self.external_id = external_id or self._id
        Pattern.__init__(self,self._id,self.source,self.external_id)
        self.address = address
        self.data = data or dict()
        self._sort_events()
        self.all_ids = set(all_ids or [])
        self._patterns = None
        self._is_leaf = is_leaf
        self.stored = stored
        # keep track of unstored nested patterns that should be stored when
        # self.store() is called
        self._to_store = set()
    
    def copy(self):
        '''
        Create an exact duplicate of this pattern with a new id.  copy() should
        always be called before modifying a stored pattern. 
        '''
        _id = self.env().newid()
        addr = None if self.address is None else self.address.copy()
        z = Zound(source = self.source,
                  external_id = _id,
                  _id = _id,
                  address = addr,
                  data = deepcopy(self.data),
                  all_ids = self.all_ids.copy(),
                  is_leaf = self.is_leaf,
                  stored = False)
        
        z._to_store = self._to_store.copy()
        return z
    
    
    def __add__(self,other):
        '''
        overlay two patterns
        '''
        if not other:
            return self.copy()
        
        rn = self.copy()
        p = other.patterns
        for k,v in other.data.iteritems():
            rn.append(p[k],v)
        return rn
    
    def __radd__(self,other):
        '''
        Implemented so sum([p1,p2,...]) can be called on a list of patterns
        '''
        if not other:
            return self.copy()
        
        return self + other
    
    # TODO: Test
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
    
    # TODO: Test
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
        _id = e.newid()
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
                return Zound(\
                        source = source,address = addr.address,is_leaf = True)
        except AttributeError:
            pass
        
        raise ValueError('addr must be a Zounds address, a Zounds frame _id, or a Frames instance with the address property set')
            
        
        
    @property
    def is_leaf(self):
        return self._is_leaf 
    
    # TODO: Tests
    # TODO: Move this into the base Pattern class
    # BUG: What if a transform changes the length of the samples?
    @property
    def length_samples(self):
        '''
        The length of this pattern in samples, when rendered as raw audio
        '''
        try:
            # this pattern has been analyzed and is in the frames database,
            # so it's trivial to find out its length in samples
            return self.env().frames_to_samples(len(self.address))
        except AttributeError:
            # this pattern hasn't yet been analyzed, so we have to calculate
            # its length in samples
            last = 0
            for k,v in self.data.iteritems():
                # get the length of the sub-pattern in samples
                l = self._patterns[k].length_samples
                st = v[-1][1] * self.env().samplerate
                total = l + st
                if total > last:
                    last = total
            
            return last
    
    # TODO: Tests
    # TODO: Move this into the base Pattern class
    @property
    def length_seconds(self):
        '''
        The length of this pattern in seconds, when rendered as raw audio
        '''
        return self.length_samples / self.env.samplerate
    
    # TODO: Tests
    def _render(self):
        # render the pattern as audio
        # KLUDGE: Maybe _render should be an iterator, for very long patterns
        
        env = self.env()
        if not self.data:
            # this is a "leaf" pattern that has already been rendered and analyzed,
            # so it can just be retrieved from the data store
            return env.synth(env.framemodel[self.address].audio)
    
        # allocate memory to hold the entire pattern
        audio = np.zeros(self.length_samples,dtype = np.float32)
        
        for k,v in self.data.iteritems():
            # render the sub-pattern
            p = self._patterns[k]
            a = p._render()
            for event in v:
                # render each occurrence of the sub-pattern
                time,tc = event[0],TransformChain.fromdict(event[1])
                ts = int(time * env.samplerate)
                # apply any transformations and add the result to the output
                audio[ts : ts + len(a)] += tc(a)
            
        return audio
    
    # TODO: Tests
    def audio_extractor(self,needs = None):
        e = self.env
        return AudioFromMemory(e.samplerate,
                               e.windowsize,
                               e.stepsize,
                               self._render(),
                               needs = needs)
    
    def _sort_events(self):
        for v in self.data.itervalues():
            v.sort()
    
    # TODO: Tests
    @property
    def patterns(self):
        
        if None is self._patterns:
            # fetch all the ids that are stored at once
            plist = self.__class__[self.all_ids - set((p._id for p in self._to_store))]
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
        
        return not bool(sum([len(v) for v in self.data.itervalues()]))
    
    
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
            l = self.data[pattern._id]
        except KeyError:
            l = []
            self.data[pattern._id] = l
        
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
    
    # TODO: Tests
    def transform(self,transform,changed = False,top = True):
        
        # TODO: Ensure that transform isn't None, and has at least one
        # transformation defined
        
        if self.is_leaf:
            n = self.__class__(source = self.source,
                               address = self.address,
                               is_leaf = self.is_leaf)
            t = transform(n, changed = changed)
            # this is a leaf pattern, and it wasn't altered in any way, so
            # return self. Otherwise, return the modified pattern
            return self if self._leaf_compare(t) else t
        
        # create a new, empty pattern
        n = self.__class__(source = self.source)
        
        for pattern,events in self.iter_patterns():
            p,e = pattern,events
            try:
                # there's a transform defined for this pattern
                p,e = transform(pattern,events,changed = changed, top = False)
                if not e:
                    # the pattern is being removed
                    continue
            except KeyError as ke:
                if not top:
                    raise ke
                
                # there was no transform defined for this pattern
                pass
            
            # append the possibly transformed events to the new pattern
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
        for k,e in self.data.iteritems():
            yield p[k],e
    
    def __iter__(self):
        yield self
    
    
    # TODO: Tests
    def __getitem__(self,key):
        raise NotImplemented()
    
    
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
        
        d['all_ids'] = self.all_ids
        d['data'] = self.data
        return d
        
    
    @classmethod
    def fromdict(cls,d,stored = False):
        
        # KLUDGE: Given the following implementation, this copy() is necessary for
        # the InMemory controller implementation, but is probably unnecessary
        # and inefficient for "real" implementations 
        d = d.copy()
        
        if d.has_key('address'):
            d['all_ids'] = None
            d['data'] = None
            d['address'] = cls.env().address_class.fromdict(d['address'])
        else:
            d['address'] = None
        
        d['stored'] = stored
        return Zound(**d)
    
    # TODO: Should this be asynchronous ?
    def store(self):
        
        if self.stored:
            # KLUDGE: Should I define a custom exception for this?
            raise Exception('Zound %s is already stored' % self._id)
    
        if self.empty:
            # KLUDGE: Should I define a custom exception for this?
            raise Exception('Cannot store an empty pattern')
        
        # store any nested unstored patterns 
        for p in self._to_store:
            p.store()
        
        # store self
        self.__class__._store(self)