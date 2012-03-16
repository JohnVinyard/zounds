from model import Model
from analyze.extractor import Extractor,ExtractorChain
from analyze.feature import \
    RawAudio,LiteralExtractor,CounterExtractor,MetaDataExtractor
from util import recurse,sort_by_lineage


# TODO: MultiFeature class (like for minhash features)
class Feature(object):
    '''
    
    '''
    def __init__(self,extractor_cls,store=True,needs=None,**kwargs):
        self.extractor_cls = extractor_cls
        self.store = store
        self.args = kwargs
        
        if not needs:
            self.needs = []
        elif isinstance(needs,list):
            self.needs = needs
        else:
            self.needs = [needs]
        
    def extractor(self,needs = None,key = None):
        '''
        Return an extractor that's able to compute this feature.
        '''
        return self.extractor_cls(needs = needs,key = key,**self.args)
    
    @recurse
    def depends_on(self):
        '''
        Return all features that this feature directly or indirectly depends
        on.
        '''
        return self.needs
    
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
    

class Precomputed(Extractor):
    '''
    Read pre-computed features from the database
    '''
    
    def __init__(self,_id,feature_name,controller,needs = None):
        Extractor.__init__(self,key=feature_name,needs=needs)
        self._c = controller
        self._id = _id
        self.stream = None
        
    @property
    def dim(self):
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
        
    def _process(self):
        '''
        Simply read this feature from the datastore until we've reached the
        stop_frame
        '''
        if None is self.stream:
            self.stream = self._c.iter_feature(self._id,self.key)
        
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
        


class MetaFrame(type):

    def __init__(self,name,bases,attrs):
        
        # Possible KLUDGE: Creating a dictionary with a subset of class
        # properties seems kind of redundant, but I'm trying to avoid
        # having to perform the search over and over.
        self.features = {}
        
        # TODO: Refactor this repeated code
        for b in bases:
            for k,v in b.__dict__.iteritems():
                if isinstance(v,Feature):
                    self.features[k] = v
        
        for k,v in attrs.items():
            if isinstance(v,Feature):
                self.features[k] = v
        
        super(MetaFrame,self).__init__(name,bases,attrs)
        
    


class Frames(Model):
    '''
    
    '''
    __metaclass__ = MetaFrame
    
    _string_dtype = 'a36'
    
    _id = Feature(LiteralExtractor,needs = None,dtype = _string_dtype)
    source = Feature(LiteralExtractor, needs = None, dtype = _string_dtype)
    external_id = Feature(LiteralExtractor, needs = None, dtype = _string_dtype)
    framen = Feature(CounterExtractor,needs = None)
    
    def __init__(self):
        Model.__init__(self)
        
    class DummyPattern:
        _id = None
        source = None
        external_id = None
        filename = None
        start_frame = None
        stop_frame = None
    
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
            chain = cls.extractor_chain(Frames.DummyPattern)
        d = {}
        env = cls.env()
        for e in chain:
            if not isinstance(e,MetaDataExtractor) and \
                (isinstance(e,RawAudio) or cls.features[e.key].store):
                d[e.key] = (e.dim(env),e.dtype,e.step)
        return d
    
    @classmethod
    def sync(cls):
        '''
        '''
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
        # BUG: If store switches from True to False, mark this as a deletion
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
        # BUG: If store switches from False to True, mark this as an add
        add = dict()
        update = dict()
        for k,v in newfeatures.iteritems():
            recompute = False
            if (k not in cls.features) or (v.store and not cls.features[k].store):
                # This is a new feature, or it 
                # has switched from un-stored to stored
                recompute = True
                add[k] = v
                
            if (k not in delete) and (k not in add) and (v != cls.features[k]):
                # The feature has changed
                recompute = True
                update[k] = v
                
            if recompute:
                torecompute.append(v)
            
        
        # do a second pass over the features. Any feature with an ancestor
        # that must be recomputed or is not stored must be re-computed
        for v in newfeatures.values():
            if v not in torecompute and\
                 any([a in torecompute or not a.store for a in v.depends_on()]):
                torecompute.append(v)      
                
            
        return add,update,delete,torecompute
    
    @classmethod
    def raw_audio_extractor(cls,pattern, needs = None):
        config = cls.env()
        ra = RawAudio(
                    config.samplerate,
                    config.windowsize,
                    config.stepsize,
                    needs = needs)
        return ra
    
    @classmethod
    def extractor_chain(cls,pattern,transitional=False,recompute = []):
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
            ra = cls.raw_audio_extractor(pattern, needs = meta)
        
            
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
                # be using RawAudio. This might not always be the case!!
                if hasattr(Frames,k):
                    needs = meta
                else:
                    needs = ra
            else:
                # this extractor depended on another feature
                needs = [d[q] for q in f.needs]
            
            
            
            if not transitional or f in recompute:
                e = f.extractor(needs=needs,key=k)
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
            