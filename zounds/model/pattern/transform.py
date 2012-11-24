from abc import ABCMeta,abstractmethod

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