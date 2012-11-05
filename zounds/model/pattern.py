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
        
    
# TODO: Add created date
# TODO: Add a changed() method, which determines whether the pattern has changed
# in any way and should be saved as a new pattern
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
    
    # TODO: Tests
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
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return tostring(self,_id = self._id,source = self.source,
                        external_id = self.external_id,all_ids = self.all_ids,
                        _is_leaf = self._is_leaf)
    
    def __hash__(self):
        return self._id.__hash__()
    
    def __eq__(self,other):
        if self is other:
            return True
        
        return (self._id == other._id) and \
               (self.source == other.source) and \
               (self.external_id == other.external_id) and \
               (self.all_ids == other.all_ids) and \
               (self._is_leaf == other._is_leaf) 
    
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
            plist = self.__class__[self.all_ids]
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
    
    # TODO: Tests
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
    
    # TODO: Tests
    def remove(self,pattern_id = None, criteria = None):
        self._check_stored()
        # TODO: be sure to remove items from all_ids and _to_store, when necessary
        raise NotImplemented()
    
    # TODO: Tests
    def transform(self):
        n = self.copy()
        # do stuff to n
        return n
    
    
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
            return d
        
        d['all_ids'] = self.all_ids
        d['data'] = self.data
        return d
        
    
    @classmethod
    def fromdict(cls,d,stored = False):
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