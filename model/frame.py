from model import Model
from analyze.extractor import ExtractorChain
from analyze.feature import RawAudio
from util import recurse,sort_by_lineage

# TODO: Write documentation
# TODO: Write tests


class Feature(object):
    '''
    '''
    def __init__(self,extractor_cls,store=True,needs=None,**kwargs):
        self.extractor_cls = extractor_cls
        self.store = store
        self.args = kwargs
        
        if not needs:
            self.needs = None
        elif isinstance(needs,list):
            self.needs = needs
        else:
            self.needs = [needs]
        
    def extractor(self,needs = None):
        '''
        '''
        return self.extractor_cls(needs = needs,**self.args)
    
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
    
    def __hash__(self):
        return hash((\
                self.__class__.__name__,
                self.store,
                frozenset(self.args.keys()),
                frozenset(self.args.values()),
                frozenset(self.needs)))
    

# TODO: MultiFeature class (like for minhash features)

class MetaFrame(type):

    def __init__(self,name,bases,attrs):
        
        # Possible KLUDGE: Creating a dictionary with a subset of class
        # properties seems kind of redundant, but I'm trying to avoid
        # having to perform the search over and over.
        self.features = {}
        for k,v in attrs.items():
            if isinstance(v,Feature):
                self.features[k] = v
        
        super(MetaFrame,self).__init__(name,bases,attrs)
        
    
    
class Frames(Model):
    '''
    '''
    __metaclass__ = MetaFrame
    
    def __init__(self):
        Model.__init__(self)
        
        
    # TODO: This should be a class method on Frames
    @classmethod
    def extractor_chain(cls,filename):
        config = cls.env().audio
        
        # Our root element will always read audio samples from
        # files on disk (for now).
        ra = RawAudio(
                filename,
                config.samplerate,
                config.windowsize,
                config.stepsize)
        
        # We now need to build the extractor chain, but we can't be sure
        # which order we'll iterate over the extractors in, so, we need
        # to sort based on dependencies
        features = cls.features.values()
        features.sort(sort_by_lineage(Feature.depends_on))
        
        # Now that the extractors have been sorted by dependency, we can
        # confidently build them, assured that extractors will always have
        # been built by the time they're needed by another extractor
        
        # start building the chain
        chain = [ra]
        # keep track of the extractors we've built, so they can be
        # passed in to the constructors of dependent extractors as
        # necessary
        d = {}
        for f in features:
            if not f.needs:
                # this was a root extractor in the context of our data model,
                # which means that it will depend directly on audio samples.
                e = f.extractor(needs = ra)
                chain.append(e)
                d[f] = e
            else:
                # this extractor depended on another feature
                e = f.extractor(needs = [d[q] for q in f.needs])
                chain.append(e)
                d[f] = e
        
            
        return ExtractorChain(chain)
            