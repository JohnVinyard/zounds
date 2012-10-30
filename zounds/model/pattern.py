'''
Pattern from the outside world, i.e., just some sound file
{
    _id : 'wholesound'self
    source : 
    external_id : 
    address : (100,200) // A PyTables address
}

A slice of a sound file
{
    _id : 'slice'
    source : 
    external_id : 
    address : (150:160)
}
A pattern using that slice
{
    _id : 'fourquarter'    
    source : 
    external_id : 
    address : None // this pattern isn't stored as frames, so it isn't searchable
    all_ids : ['slice'],
    data : {
        'slice' : [
            (0, [{'amp' : (1,)}]),
            (1, [{'amp' : (.5,)}]),
            (2, [{'amp' : (1,)}]),
            (3, [{'amp' : (.5,)}]),
        ]
    }
}
A pattern using the previous pattern
{
    _id : 'fourmeasure',
    source : 
    external_id : 
    address : (1000,2000) // This pattern is stored, so we can just play it outright
    all_ids : ['fourquarter','slice']
    data : {
        'fourquarter' : [(0,{'amp' : .9}),
                         (1,{'amp' : .8})],
                         
        'slice'       : [(0.1,{'amp' : .01})]
    }
}
'''
import numpy as np

from zounds.model.model import Model
from zounds.analyze.feature.rawaudio import AudioFromDisk,AudioFromMemory
from zounds.analyze.synthesize import transformers,TransformChain

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
class Zound2(Pattern):
    
    __metaclass__ = MetaZound2
    
    def __init__(self,source = None,external_id = None,_id = None,
                 address = None,data = None,all_ids = None,
                 bpm = None):
        
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
        
        # TODO: Hash the pattern somehow, so it can be compared to ... itself?
        self.address = address
        self.data = data or dict()
        self._sort_events()
        self.all_ids = set(all_ids) or set()
        self._patterns = None
        
        self.bpm = bpm
        
        self.env = self.env()
        
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