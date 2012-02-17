from model import Model

class MetaPipeline(Model):
    
    @property
    def controller(self):
        return config.data[self]

# TODO: Should a pipeline be Extractor-derived?
class Pipeline(object):
    '''
    '''
    __metaclass__ = MetaPipeline
    
    def __init__(self,fetch,source,preprocess,learner):
        object.__init__(self)
        
        # e.g. Pipeline( RowModel.Bark, db, MeanStd, LinearRbm ) for a 
        # pipeline whose training data is drawn from an already existing
        # database
        
        # or Pipeline(RowModel.Bark, '/home/john/snd/FreeSound', MeanStd, LinearRbm)
        # for a pipeline whose training data is drawn from disk
        
        # Fetch should simply be a feature, like bark bands, or rbm activations.
        # If we know what our current FrameModel is, we can get samples from
        # disk or the db.  Additionally, if fetching from disk, we can only
        # run the branches of the extractor that are necessary.
        self.fetch = fetch
        self.source = source
        
        
        # something that knows how to preprocess data 
        self.preprocess = preprocess
        
        # something that knows how to train on data, and can
        # describe future data based on that training.
        self.learner = learner
        
        # example 1: an rbm that trains on bark bands
        #   - fetch grabs bark bands from disk
        #   - preprocess does mean and std regularization
        #   - the rbm learns and then can output features
        
        # example 2: minhash of rbm activations
        #   - no fetcher
        #   - no preprocessor
        #   - the "training" stage just consists of picking the
        #     hash functions (permutations), and saving them
        
        # example 3: an rbm that trains on 
        # random samples from self-similarity matrices
        #
        #   - fetcher that remembers coordinates of points
        #   - no preprocessor
        #   - rbm
        
        # Wishlist : Multiple pipelines can be chained together
        
    def save(self):
        pass
    
    def train(self):
        data = self.fetch()
        data = self.preprocess(data)
        self.save()
        # TODO checkpoints, incremental save
        self.learner.train(data)
        self.save()
        
    
    def activate(self,data):
        data = self.preprocess(data)
        return self.learner(data)
    
import config