'''



Pattern from the outside world, i.e., just some sound file
{
    _id : 'wholesound'
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
            (0, {'amp' : 1}),
            (1, {'amp' : .5}),
            (2, {'amp' : 1}),
            (3, {'amp' : .5})
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
    
        

class Zound2(Pattern):
    
    __metaclass__ = MetaZound2
    
    def __init__(self,source = None,external_id = None,_id = None,
                 address = None,data = None,all_ids = None):
        
        self.source = source or self.env().source 
        self._id = _id or self.env().newid()
        self.external_id = external_id or self._id
        Pattern.__init__(_id,source,external_id)
        if address and data:
            raise ValueError('specify one of address or data')
        self.address = address
        self.data = data or dict()
        self.all_ids = set(all_ids) or set()
    
    def _render(self):
        # render the pattern as audio
        pass
    
    def _hydrate(self):
        # fetch all nested patterns
        pass
    
    def append(self,_id,events):
        '''
        :param _id: the _id of the pattern to be added
        
        :param events: a list of two-tuples of (time_secs,transformations)
        '''
        try:
            l = self.data[_id]
        except KeyError:
            l = []
            self.data[_id] = l
        
        l.extend(events)
        self.all_ids.add(_id)
    
    def todict(self):
        d = {
             '_id' : self._id,
             'source' : self.source,
             'external_id' : self.external_id
             }
        
        if self.address:
            d['address'] = self.address.todict()
            return d
        
        d['all_ids'] = self.all_ids
        d['data'] = self.data
        return d
        
    
    @staticmethod
    def fromdict(self,d):
        if d.has_key('address'):
            d['all_ids'] = None
            d['data'] = None
        else:
            d['address'] = None
        
        return Zound2(**d)
        
    
    

        
class Zound(Pattern,list):
    
    
    
    class Event(object):
        
        def __init__(self,time_secs,data):
            '''
            time_secs is the time at which this event occurs
            
            data may be an address, or the id of another pattern. 
            '''
            self.time_secs = np.array(time_secs)
            self._time_samples = self.time_secs * self.env().sample_rate
            self.pattern = data
    
    def __init__(self,_id,source,external_id, address = None):
        Pattern.__init__(self,_id,source,external_id)
        list.__init__(self)
        # the frames backend-specific address of this pattern, if it has
        # been stored as a contiguous block in the frames database
        self.address = address
        self.created_date = None
        
        self._changed = False
    
    # TODO: An audio from db extractor
    def audio_extractor(self, needs = None):
        e = self.__class__.env()
        self._data['samples'] = self.render()
        return AudioFromMemory(e.samplerate,
                               e.windowsize,
                               e.stepsize,
                               needs = needs)
    
        
    def add(self,time_secs,data):
        try:
            list.extend(self,[Zound.Event(t,data) for t in time_secs])
        except TypeError:
            list.append(self,Zound.Event(time_secs,data))
    
    def _should_store_frames(self):
        '''
        True if this pattern should be persisted to the frames database because
        it isn't contiguous, or contains more than one event.
        '''
        return len(self) > 1
    
    # TODO: Move this into the base Pattern class
    def length_samples(self):
        '''
        The length of this pattern in samples, when rendered as raw audio
        '''
        raise NotImplemented()
    
    # TODO: Move this into the base Pattern class
    def length_seconds(self):
        '''
        The length of this pattern in seconds, when rendered as raw audio
        '''
        raise NotImplemented()
    
    # TODO: Move this into the base Pattern class
    def render(self):
        '''
        Returns a numpy array of audio samples
        '''
        # If address isn't None, then we can just read a contiguous block and
        # play it.
        
        # If address is None, then we have to follow pattern "links" to 
        # reconstruct everything
        raise NotImplemented()
    
    
        
