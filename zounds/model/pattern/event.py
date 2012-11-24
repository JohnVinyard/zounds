from __future__ import division
from copy import deepcopy
from zounds.analyze.synthesize import Transform
from zounds.util import tostring

class Event(object):
    
    def __init__(self,time,*args):
        '''__init__
        
        :param time_secs: The time in seconds at which the event should occur
        
        :param args: A list of zounds.analyze.synthesize.Transform \
        derived instances
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
        return {'time' : event.time,
                'transforms' : [t.todict() for t in event.transforms]}
    
    @staticmethod
    def decode_custom(doc):
        # TODO: This should include transform data too
        return Event(doc['time'],
                     *[Transform.fromdict(t) for t in doc['transforms']])
    
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
