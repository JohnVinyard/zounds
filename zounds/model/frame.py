from __future__ import division
from abc import ABCMeta,abstractmethod
import numpy as np

from zounds.model.model import Model
from zounds.model.pattern import FilePattern,DataPattern
from zounds.analyze.extractor import Extractor,ExtractorChain
from zounds.analyze.feature.metadata import \
    MetaDataExtractor, LiteralExtractor,CounterExtractor
from zounds.analyze.feature.rawaudio import AudioSamples
from zounds.util import recurse,sort_by_lineage
from zounds.environment import Environment


class Feature(object):
    '''
    
    '''
    def __init__(self,extractor_cls,store=True,needs=None,**kwargs):
        '''
        A useful descriptor for audio
        
        :param extractor_cls: Extractor-derived class that will compute this feature
        '''
        self.extractor_cls = extractor_cls
        self.store = store
        self.args = kwargs
        
        if not needs:
            self.needs = []
        elif isinstance(needs,list):
            self.needs = needs
        else:
            self.needs = [needs]
        
        self.key = None
        
    def extractor(self,needs = None,key = None):
        '''
        Return an extractor that's able to compute this feature.
        '''
        # KLUDGE: I'm doing this because SingleInput-derived extractors
        # cannot think they're root.  The purpose of calling this method
        # is almost certainly to investigate the data type and output dimension
        # of the extractor, so feed a dummy extractor in here.
        if not needs:
            needs = CounterExtractor()
        return self.extractor_cls(needs = needs,key = key,**self.args)
    
    @property
    def step(self):
        return self.extractor().step
    
    @property
    def nframes(self):
        return self.extractor().nframes
    
    # BUG: This may not always return the correct answer, if the property
    # depends on the input sources. See Composite extractor, e.g.
    @property
    def dtype(self):
        return self.extractor().dtype
    
    # BUG: This may not always return the correct answer, if the property
    # depends on the input sources. See Composite extractor, e.g.
    @property
    def dim(self):
        return self.extractor().dim(Environment.instance)
    
    @recurse
    def depends_on(self):
        '''
        Return all features that this feature directly or indirectly depends
        on.
        '''
        return self.needs
    
    def _stat(self,aggregate,step = None,axis = 0):
        s = self.step if step is None else step
        return Environment.instance.framecontroller.stat(\
                                            self,aggregate,step = s, axis = axis)
    
    def mean(self,step = None,axis = 0):
        return self._stat(np.mean,step = step, axis = axis)
    
    def sum(self,step = None, axis = 0):
        return self._stat(np.sum,step = step, axis = axis)
    
    def std(self,step = None, axis = 0):
        return self._stat(np.std, step = step, axis = axis)
    
    def max(self,step = None, axis = 0):
        return self._stat(np.max, step = step, axis = axis)
    
    def min(self,step = None, axis = 0):
        return self._stat(np.min, step = step, axis = axis)
    
    # TODO: Write tests
    def __eq__(self,other):
        return self.__class__ == other.__class__ \
        and self.extractor_cls == other.extractor_cls \
        and self.store == other.store \
        and self.args == other.args \
        and set(self.needs) == set(other.needs)
    
    def __ne__(self,other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((\
                self.__class__.__name__,
                self.store,
                frozenset(self.args.keys()),
                frozenset(self.args.values()),
                frozenset(self.needs if self.needs else [])))
    
    def __repr__(self):
        return '%s(%s, store = %s, needs = %s, args = %s)' %\
         (self.__class__.__name__,
          self.extractor_cls.__name__,
          self.store,
          self.needs,
          self.args)
    



class Dummy(Extractor):
    '''
    A no-op extractor, useful during database synchronization in cases where
    an unstored feature upon which stored features depend has no changes in its
    lineage.  This means that the depdendent features can simply be read from 
    the database, however, we'd like to insert a proxy for the unstored feature,
    to keep extractor chain construction simple and easy to understand.
    '''
    def __init__(self,_id,feature_name,controller, needs = None):
        Extractor.__init__(self, key = feature_name, needs = needs)
        self._c = controller
        self._id = _id
        self.stream = None
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _init_stream(self):
        self.stream = xrange(len(self._c[self._id])).__iter__()

    def _process(self):
        if None is self.stream:
            self._init_stream()
        try:
            return self.stream.next()
        except StopIteration:
            self.out = None
            self.done = True
            if self.sources:
                self.sources[0].done = True
                self.sources[0].out = None
    
    def __hash__(self):
        return hash(\
                    (self.__class__.__name__,
                     self._id,
                     self.key))

            
class Precomputed(Dummy):
    '''
    Read pre-computed features from the database
    '''
    
    
    def __init__(self,_id,feature_name,controller,needs = None):
        Dummy.__init__(self,_id,feature_name,controller, needs = needs)
        
    
    def dim(self,env):
        '''
        Ask the datastore about the dimension of this data
        '''
        return self._c.get_dim(self.key)
    
    @property
    def dtype(self):
        '''
        Ask the datastore about the type of this data
        '''
        return self._c.get_dtype(self.key)
    
    def _init_stream(self):
        if 'audio' != self.key:
            features = self._c.get_features()
            # get an extractor for this feature
            extractor = features[self.key].extractor()
            # ask the extractor about the stepsize and frame length 
            # for this feature
            self.step = extractor.step
            self.nframes = extractor.nframes
        # get an iterator that will iterate over this feature with 
        # the step size we just discovered
        self.stream = self._c.iter_feature(\
                                self._id,self.key,step = self.step)

# TODO: Write better documentation
class Address(object):
    '''
    A container for the most efficient way to access frames, given the backing
    store.
    ''' 
    __metaclass__ = ABCMeta
    
    def __init__(self,key):
        '''
        key is the location of one or more frames, addressed in a manner 
        suitable for the Frames backing-store
        '''
        object.__init__(self)
        self._key = key
    
    @property
    def key(self):
        return self._key
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def serialize(self):
        pass
    
    @classmethod
    def deserialize(cls):
        raise NotImplemented()
    
    @abstractmethod
    def __len__(self):
        '''
        The number of frames represented by this address
        '''
        pass
    
    @classmethod
    def congeal(cls,addresses):
        '''
        Given many addresses, return one contiguous address that contains
        them all.
        '''
        pass
    
    @abstractmethod
    def __eq__(self,other):
        pass
    
    @abstractmethod
    def __ne__(self,other):
        pass
    
    @abstractmethod
    def __lt__(self,other):
        pass
    
    @abstractmethod
    def __le__(self,other):
        pass
    
    @abstractmethod
    def __gt__(self,other):
        pass
    
    @abstractmethod
    def __ge__(self,other):
        pass
    
    @abstractmethod
    def __hash__(self):
        pass
    

class MetaFrame(type):

    def __init__(self,name,bases,attrs):
        
        self.features = {}
        
        for b in bases:
            self._add_features(b.__dict__)
        
        self._add_features(attrs)
        
        super(MetaFrame,self).__init__(name,bases,attrs)
        
        
    def _add_features(self,d):
        for k,v in d.iteritems():
            if isinstance(v,Feature):
                v.key = k
                self.features[k] = v
    
    def iterfeatures(self):
        return self.features.iteritems()
    
        
    
    def __getitem__(self,address):
        '''
        key may be one of the following:
        
        - a zounds id
        - a two-tuple of (source,external_id)
        - a frame address
        '''
        return self(address)


class Frames(Model):
    '''
    
    '''
    __metaclass__ = MetaFrame
    
    _string_dtype = 'a32'
    
    _id = Feature(LiteralExtractor,needs = None,dtype = _string_dtype)
    source = Feature(LiteralExtractor, needs = None, dtype = _string_dtype)
    external_id = Feature(LiteralExtractor, needs = None, dtype = _string_dtype)
    framen = Feature(CounterExtractor,needs = None)
    
    
    def __init__(self,address = None, data = None):
        Model.__init__(self)
        if None is not address:
            self.address = address
            self._data = self.controller()[address]
            if not len(self._data):
                print address
                raise KeyError(address)
        elif None is not data:
            self._data = data
        else:
            raise ValueError('address or data must be supplied')
        
        
        self.audio = self._data['audio'] 
        
        for k,v in self.__class__.stored_features().iteritems():    
            setattr(self,k,self._data[k])
            
     
    def __len__(self):
        return len(self._data)
    
    # TODO: Write tests for overlapping and non-overlapping windows
    @property
    def seconds(self):
        return self.env().frames_to_seconds(len(self))
    
    def __getitem__(self,key):
        if isinstance(key,str):
            try:
                return getattr(self,key)
            except (AttributeError,ValueError):
                # for some reason, numpy recarrays raise a ValueError instead
                # ok a key error when a field is not present
                raise KeyError(key)
        elif isinstance(key,Feature):
            if key.key is None:
                raise KeyError(None)
            try:
                return getattr(self,key.key)
            except (AttributeError,ValueError):
                # This probably means that the feature hasn't been wired-up
                # by the MetaFrame class and doesn't have its 'key' property
                # set. Treat this as a KeyError
                raise KeyError(key)
        elif isinstance(key,int):
            # TODO: Should this return a numpy.recarray or another 
            # Frames instance?
            return self.__class__(data = self._data[key:key+1])
        elif isinstance(key,slice):
            # TODO: Should this return a numpy.recarray or another 
            # Frames instance?
            return self.__class__(data = self._data[key])
        elif isinstance(key,list) or isinstance(key,np.ndarray):
            return self.__class__(data = self._data[key])
        else:
            raise ValueError('key must be a string, Feature, int, or slice')
    
    def __getattribute__(self,k):
        '''
        This method is implemented so that non-stored features can be computed
        on the fly, if requested.
        '''
        f = Model.__getattribute__(self,k)
        if isinstance(f,Feature):
            # synthesize audio from this list of frames
            audio = self.__class__.env().synth(self.audio)
            p = DataPattern(None,None,None,audio)
            # create an extractor chain that includes only the extractors we'll
            # need to compute the requested feature
            ec = self.__class__.extractor_chain(p).prune(k)
            # run the chain
            d = ec.collect()
            # cache the results
            setattr(self,k,np.array(d[k]).squeeze())
            return Model.__getattribute__(self,k)
        else:
            return f
    
    @classmethod
    def random(cls):
        '''
        Return the frames of a random pattern/sound
        '''
        _ids = list(cls.list_ids())
        return cls[_ids[np.random.randint(0,len(_ids) - 1)]]
        
    @classmethod
    def list_ids(cls):
        return cls.controller().list_ids()
    
    @classmethod
    def stored_features(cls):
        '''
        Return a dictionary containing the subset of stored features
        '''
        return dict((k,v) for k,v in cls.features.iteritems() if v.store)
        
    @classmethod
    def dimensions(cls,chain = None):
        '''
        Return a dictionary mapping feature keys to three-tuples of 
        (shape,dtype,stepsize)
        '''
        # KLUDGE: I have to pass a pattern to build an extractor chain,
        # but I'm building it here to learn about the properties of the
        # extractors, and not to do any real processing.
        if not chain:
            chain = cls.extractor_chain(FilePattern(None,None,None,None))
        d = {}
        env = cls.env()
        for e in chain:
            if not isinstance(e,MetaDataExtractor) and \
                (isinstance(e,AudioSamples) or cls.features[e.key].store):
                d[e.key] = (e.dim(env),e.dtype,e.step)
        return d
    
    @classmethod
    def _sync(cls):
        c = cls.controller()
        features = c.get_features()
        # create a class using features that are currently in the database
        class OldModel(Frames):
            pass
        
        for k,v in features.iteritems():
            setattr(OldModel,k,v)
            OldModel.features[k] = v
        
        # create an update plan
        add,update,delete,recompute = OldModel.update_report(cls)
        return add,update,delete,recompute
    
    @classmethod
    def sync(cls):
        '''
        '''
        
        add,update,delete,recompute = cls._sync()
        c = cls.controller()
        if any([add,update,delete,recompute]):
            # something has changed. Sync the database
            c.sync(add,update,delete,recompute)
        
         
    @classmethod
    def update_report(cls,newframesmodel):
        '''
        Compare new and old versions of a Frames-derived class and produce a 
        "report" about which features have been added,updated or deleted. 
        '''
        newfeatures = newframesmodel.features
        torecompute = []
        
        # figure out which features will be deleted
        delete = dict()
        for k,v in cls.features.iteritems():
            if k not in newfeatures:
                # this feature isn't in the new FrameModel class.
                # Record the fact that it'll need to be deleted.
                delete[k] = v
                continue
            
            if not newfeatures[k].store and v.store:
                delete[k] = v
        
        
        
        # do a pass over the features, recording features that are new
        # or have been changed
        add = dict()
        update = dict()
        for k,v in newfeatures.iteritems():
            if (k not in cls.features) or (v.store and not cls.features[k].store):
                # This is a new feature, or it 
                # has switched from un-stored to stored
                add[k] = v
                torecompute.append(v)
                continue
                
            if (k not in delete) and (k not in add) and (v != cls.features[k]):
                # The feature has changed
                update[k] = v
                torecompute.append(v)
                continue
                
        
        toregenerate = []
        # do a second pass over the features. Any feature with an ancestor
        # that must be re-computed must also be re-computed 
        for v in newfeatures.itervalues():
            depends_on = v.depends_on()
            
            if v not in torecompute and\
                 any([a in torecompute for a in depends_on]):
                # this feature has at least one ancestor that also 
                # needs to be re-computed
                torecompute.append(v)
            
            # handle special cases where the ancestor of a recomputed feature
            # is not stored. This means that we have to recompute the ancestor
            # and the recomputed feature, but other features that rely on the
            # ancestor needn't be recomputed.  Store these in a different list,
            # for now, so they don't trigger recomputes on all descendants of
            # the ancestor. Ugh. 
            for d in depends_on:
                if v in torecompute and not d.store and d not in torecompute:
                    toregenerate.append(d)
        
        
        return add,update,delete,torecompute + toregenerate
    

    @classmethod
    def extractor_chain(cls,
                        pattern,
                        transitional=False,
                        recompute = []):
        '''
        From the features defined on this class, create a feature extractor
        that can transform raw audio data into the desired set of features.
        '''
        
        meta = MetaDataExtractor(pattern,key = 'meta') 
        if transitional:      
            ra = Precomputed(pattern._id,
                             'audio',
                             cls.controller(),
                             needs = meta)
        else:
            ra = pattern.audio_extractor(needs = meta)
        
            
        # We now need to build the extractor chain, but we can't be sure
        # which order we'll iterate over the extractors in, so, we need
        # to sort based on dependencies
        features = cls.features.items()
        by_lineage = sort_by_lineage(Feature.depends_on)
        srt = lambda lhs,rhs : by_lineage(lhs[1],rhs[1])
        features.sort(srt)
        
        
        # Now that the extractors have been sorted by dependency, we can
        # confidently build them, assured that extractors will always have
        # been built by the time they're needed by another extractor
        
        # start building the chain
        chain = [meta,ra]
        # keep track of the extractors we've built, so they can be
        # passed in to the 'needs' parameter of the constructors of dependent 
        # extractors as necessary
        d = {}
        for k,f in features:
            if not f.needs:
                # this was a root extractor in the context of our data model,
                # which means that it will depend directly on audio samples.
                
                # KLUDGE: I'm assuming that any Features defined on the base
                # Frames class will be using the MetaDataExtractor as a source,
                # while any features defined on the Frames-derived class will
                # be using AudioFromDisk. This might not always be the case!!
                if hasattr(Frames,k):
                    needs = meta
                else:
                    needs = ra
            else:
                # this extractor depended on another feature
                needs = [d[q] for q in f.needs]
                
            
            
            
            if not transitional or f in recompute:
                e = f.extractor(needs=needs,key=k)
            elif not f.store and k not in recompute:
                # We're in a situation where the old feature doesn't 
                # need to be recomputed, but also isn't stored. This means that
                # the precomputed extractor would try to read values that aren't
                # in the database.  Insert a dummy extractor.
                e = Dummy(pattern._id,k,cls.controller())
            else:
                # Nothing in this feature's lineage has changed, so
                # we can safely just read values from the database
                e = Precomputed(pattern._id,
                                k,
                                cls.controller(),
                                needs = needs)
            chain.append(e)
            d[f] = e
            

        return ExtractorChain(chain)