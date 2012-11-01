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


import numpy as np

from zounds.model.model import Model
from zounds.analyze.feature.rawaudio import AudioFromDisk,AudioFromMemory
from zounds.analyze.synthesize import transformers,TransformChain
from zounds.environment import Environment

class Pattern(Model):
    '''
    A Pattern is the central concept in zounds.  Patterns can be nested.
    a "leaf" pattern represents a list of ids which point to audio frames.
    A "branch" pattern points to other patterns.
    '''
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

class MetaZound2(type):
    
    def __init__(self,name,bases,attrs):
        super(MetaZound2,self).__init__(name,bases,attrs)
    
    def __getitem__(self,key):
        return self.fromdict(self.controller()[key])
    
    def store(self):
        self.controller().store(self.todict())
    
        
# KLUDGE: What if I alter the volume of an event slightly?  Does that warrant
# saving a "copy" of the pattern?

# TODO: Add created date
# TODO: Add a changed() method, which determines whether the pattern has changed
# in any way and should be saved as a new pattern
class Zound2(Pattern):
    
    __metaclass__ = MetaZound2
    
    env = Environment.instance
    
    def __init__(self,source = None,external_id = None,_id = None,
                 address = None,data = None,all_ids = None,is_leaf = False):
        
        # source is the name of the application or user that created this pattern
        self.source = source or self.env().source
        # _id is the zounds _id of this pattern
        self._id = _id or self.env().newid()
        # external_id is the _id assigned to the pattern by the application or
        # user that created it
        self.external_id = external_id or self._id
        Pattern.__init__(_id,source,external_id)
        
        
        if not address:
            raise ValueError('specify one of address or data')
        

        self.address = address
        self.data = data or dict()
        self._sort_events()
        self.all_ids = set(all_ids) or set()
        self._patterns = None
        self._is_leaf = is_leaf
        
        self.env = self.env()
    
    @classmethod
    def leaf(cls,addr,source = None):
        '''leaf
        
        :param addr: Address can be a Zounds frame id, a Zounds frame address, or \
        a Frames-derived instance with the address property set
        '''
        _id = cls.env.newid()
        try:
            a = cls.env.address_class(addr)
        except ValueError:
            pass
        
        try:
            a = cls.env.framecontroller.address(addr)
        except KeyError:
            pass
        
        try:
            a = addr.address
        except AttributeError:
            raise ValueError('addr must be a Zounds address, a Zounds frame _id, or a Frames instance with the address property set')
        
        source = source or cls.env.source
        return Zound2(source = source,address = a,is_leaf = True)
            
        
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
            return self.env.frames_to_samples(len(self.address))
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
    
    # TODO: Move this into the base Pattern class
    @property
    def length_seconds(self):
        '''
        The length of this pattern in seconds, when rendered as raw audio
        '''
        return self.length_samples / self.env.samplerate
    
    
    def _render(self):
        # render the pattern as audio
        # KLUDGE: Maybe _render should be an iterator, for very long patterns
        
        env = self.env
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
    
    @property
    def patterns(self):
        
        if None is self._patterns:
            self._patterns = self.__class__[self.all_ids]
            return self._patterns
        
        
        for _id in self.all_ids:
            if not self._patterns.has_key(_id):
                self._patterns[_id] = self.__class__[_id]
        
        return self._patterns
    
    
    
    def append(self,pattern,events):
        '''append
        
        Add a pattern at one or more locations in time to this pattern
        
        :param _id: the _id of the pattern to be added
        
        :param events: a list of two-tuples of (time_secs,transformations)
        '''
        if pattern._id == self._id:
            raise ValueError('Patterns cannot contain themselves!')
        
        try:
            l = self.data[pattern._id]
        except KeyError:
            l = []
            self.data[pattern._id] = l
        
        l.extend(events)
        l.sort()
        self.all_ids.add(pattern._id)
        self.all_ids.update(pattern.all_ids)
    
    def todict(self):
        d = {
             '_id' : self._id,
             'source' : self.source,
             'external_id' : self.external_id
             }
        
        if self.bpm:
            d['bpm'] = self.bpm
        
        if self.address:
            d['address'] = self.address.todict()
            return d
        
        d['all_ids'] = self.all_ids
        d['data'] = self.data
        return d
        
    
    @classmethod
    def fromdict(cls,d):
        if d.has_key('address'):
            d['all_ids'] = None
            d['data'] = None
            d['address'] = cls.env.address_class.fromdict(d['address'])
        else:
            d['address'] = None
        
        return Zound2(**d)