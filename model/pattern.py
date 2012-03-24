'''
A leaf pattern represents a block of contiguous, ascending (in time) frames in the frames
database.

A branch pattern contains "pointers" to leaf patterns (or contiguous subsets of them), 
or other branch patterns.

An audio_event, in the parlance of zounds 1, is a branch pattern that points at
{_id : 1234, source : 'FreeSound', external_id : '9'}[100:200], e.g.. This means
 that there can be pattern ids in the patterns database that don't exist in the
 frames database.

If a user creates a non-contiguous leaf pattern, this pattern should be appended
to the frames database so that it becomes contiguous, e.g., the pattern 
[10,30,1,16] would be appended to the end of the frames database with a new id,
a source equal to the user's name, a new external id, and an address equal to
[original_db_length : original_db_length + 4]

There needs to be a level of indirection regarding the address space. Row numbers
might not be appropriate to every conceivable backing store.

BUG: Consider a branch pattern whose phenotype is one measure of the same hihat
sound playing eighth-notes.  This pattern is two-deep. It consists of a single
branch pattern, repeated, which points at a slice of an original, longer sound.
If it is to be a candidate for similarity searches, it must be analyzed and saved
to the frames database.

BUG: What if I edit the aforementioned pattern so that its length changes, or
I delete it?  That means that the addresses of all patterns following it in the
frames database also change, which means all patterns following it must be updated.

BUG: What if I make small changes to a pattern? Must it be re-stored?  What 
about patterns that reference it?  The dumb solution seems to be to always
re-store it, so that I don't have to worry about updating pattern references,
but won't this lead to a very large frames database full of redundant info?

BUG: Consider the last, musical pattern below. What if I simply repeated it
four times (ie, four measures of four quarter notes each)?  Would it be 
re-stored?

// Leaf Pattern
leaf = {
    _id         : leaf_uuid,
    source      : 'FreeSound',
    external_id : '9',
    ancestors   : (),
    address     : (start,stop) OR (_id)
}

// Audio-Event-like pattern. This would not be re-stored, since it represents
// a single occurrence of a block of contiguous, ascending frames
ae = {
    _id         : branch_uuid,
    source      : 'John',
    external_id : other_uuid,
    ancestors   : (leaf_uuid)
    address     : (start,stop) 
}

// Musical Patern. This would be re-stored, since it represents multiple 
// occurrences of a contiguous block.
m = {
    _id         : music_uuid,
    source      : 'John',
    external_id : other_uuid,
    ancestors   : (leaf_uuid,branch_uuid),
    address     : (start,stop),
    pattern     : [
        {
            time : [0,1,2,3],
            _id  : branch_uuid
        }
    ] 
}
'''
import numpy as np

from model import Model
from analyze.feature import AudioFromDisk,AudioFromMemory


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
        self._data['filename'] = self.filename

    
    def audio_extractor(self, needs = None):
        e = self.__class__.env()
        return AudioFromDisk(e.samplerate,
                             e.windowsize,
                             e.stepsize,
                             needs = needs)

class DataPattern(Pattern):
    '''
    Represents a pattern in the form of an in-memory numpy array of audio 
    samples that has not yet been stored
    '''
    def __init__(self,_id,source,external_id,samples):
        Pattern.__init__(self,_id,source,external_id)
        self.samples = samples
        self._data['samples'] = samples
        
    
    def audio_extractor(self,needs = None):
        e = self.__class__.env()
        return AudioFromMemory(e.samplerate,
                               e.windowsize,
                               e.stepsize,
                               needs = needs)

class Zound(Pattern,list):
    
    class Event(object):
        
        def __init__(self,time_secs,data):
            '''
            time_secs can be a single time, or a list of them. A list implies
            that data occurs multiple times
            
            data may be an address, or the id of another pattern. 
            '''
            self.time_secs = np.array(time_secs)
            self._time_samples = self.time_secs * self.env().sample_rate
            self.pattern = data
    
    def __init__(self,_id,source,external_id):
        Pattern.__init__(self,_id,source,external_id)
        list.__init__(self)
        self._ancestors = []
        
    def add(self,time_secs,data):
        try:
            list.extend(self,Zound.Event(t,data) for t in time_secs)
        except TypeError:
            list.append(self,Zound.Event(time_secs,data))
    
    def _should_store_frames(self):
        '''
        True if this pattern should be persisted to the frames database because
        it isn't contigous, or contains more than one event.
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
        raise NotImplemented()
    
    
        
