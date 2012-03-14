from model import Model
from analyze.extractor import Extractor,ExtractorChain
from analyze.feature import \
    RawAudio,LiteralExtractor,CounterExtractor,MetaDataExtractor
from util import recurse,sort_by_lineage

# TODO: Write documentation
# TODO: Write tests

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
        '''
        return self.extractor_cls(needs = needs,key = key,**self.args)
    
    @recurse
    def depends_on(self):
        '''
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
    #BUG: This won't work. A span of frames should
    # be supplied!!!  This will always start reading
    # from the beginning of the db!
    def __init__(self,feature_name,controller):
        Extractor.__init__(self,key=feature_name)
        self._c = controller
        self._frame = 0
        
    @property
    def dim(self):
        return self._c.get_dim(self.key)
    
    @property
    def dtype(self):
        return self._c.get_dtype(self.key)
        
    def _process(self):
        data = self._c.get(self._frame,self.key)
        self._frame += 1
        return data
    
    def __hash__(self):
        return hash(\
                    (self.__class__.__name__,
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
    
    # TODO: Factor out 36-length string dtype here
    _id = Feature(LiteralExtractor,needs = None,dtype = 'a36')
    source = Feature(LiteralExtractor, needs = None, dtype = 'a36')
    external_id = Feature(LiteralExtractor, needs = None, dtype = 'a36')
    #filename = Feature(LiteralExtractor, needs = None, store = False, dtype = 'a36')
    framen = Feature(CounterExtractor,needs = None)
    
    '''
    When creating a plan to transition from one FrameModel to the next, this
    flag is used to mark features that will need to be recomputed because
    they're new, they've changed, or an extractor somewhere in their lineage
    has changed.
    '''
    recompute_flag = '_recompute'
    
    def __init__(self):
        Model.__init__(self)
        
        # KLUDGE: These shouldn't be special. They should be defined just like
        # other features are
        self.source = None
        self._id = None
        self.framen = None
    
    class DummyPattern:
        _id = None
        source = None
        external_id = None
        filename = None
        
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
            chain = cls.extractor_chain(pattern = Frames.DummyPattern)
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
        # TODO: Chain should be a function that can create an appropriate
        # extractor chain that will process only a certain span of frames
        add,update,delete,chain = OldModel.update_report(cls)
        
        if add or update or delete:
            c.sync(add,update,delete,chain)
        
        c.set_features(cls.features)
         
    @classmethod
    def update_report(cls,newframesmodel):
        '''
        '''
        newfeatures = newframesmodel.features
        
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
            setattr(v,cls.recompute_flag,recompute)
        
        # do a second pass over the features. Any feature with an ancestor
        # that must be recomputed or is not stored must be re-computed
        for v in newfeatures.values():
            if not v._recompute:
                v._recompute = any([a._recompute or not a.store for a in v.depends_on()])     
            
            
        return add,\
            update,\
            delete,\
            newframesmodel.extractor_chain(transitional=True)
    
    @classmethod
    def root_extractor(cls,pattern):
        config = cls.env()
        meta = MetaDataExtractor(pattern,key = 'meta')
        ra = RawAudio(
                    config.samplerate,
                    config.windowsize,
                    config.stepsize,
                    needs = meta)
        return meta,ra
    
    @classmethod
    def extractor_chain(cls,pattern = None,transitional=False):
        '''
        '''
        if (pattern and transitional) or (not pattern and not transitional):
            # Neither or both parameters were supplied. One or the other 
            # is required
            raise ValueError('Either pattern or transitional must be supplied')
        
        if pattern:
            meta,ra = cls.root_extractor(pattern)
        else:
            # BUG: This won't work anymore now that I've introduced the metadata
            # extractor
            if not all([hasattr(f,cls.recompute_flag) \
                        for f in cls.features.values()]):
                            raise ValueError('A call to update_report is necessary\
                            prior to creating a transitional extractor')
                    
            ra = Precomputed('audio',cls.controller())
        
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
        # passed in to the constructors of dependent extractors as
        # necessary
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
                
            if not hasattr(f,cls.recompute_flag) or f._recompute:
                e = f.extractor(needs=needs,key=k)
            else:
                # Nothing in this feature's lineage has changed, so
                # we can safely just read values from the database
                e = Precomputed(k,cls.controller())
            chain.append(e)
            d[f] = e
            
            
        return ExtractorChain(chain)
            