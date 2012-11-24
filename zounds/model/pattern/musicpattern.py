from __future__ import division
from copy import deepcopy

from zound import Zound
from event import Event
from transform import RecursiveTransform

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
            
            # BUG: What if *self* has negative or wrapped event times? The same
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