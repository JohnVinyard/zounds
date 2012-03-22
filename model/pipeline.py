from model import Model



# TODO: Should a pipeline be Extractor-derived?
class Pipeline(Model):
    '''
    '''
    
    def __init__(self,_id,fetch,preprocess,learner):
        Model.__init__(self)
        
        # e.g. Pipeline( RowModel.Bark, db, MeanStd, LinearRbm ) for a 
        # pipeline whose training data is drawn from an already existing
        # database
        
        # or Pipeline(RowModel.Bark, '/home/john/snd/FreeSound', MeanStd, LinearRbm)
        # for a pipeline whose training data is drawn from disk
        
        # a unique identifier
        self._id = _id
        
        # For now, I'm only going to implement a fetcher that knows how to read
        # features from a frames db.  Reading stuff from disk was always very
        # slow, so, for now, it's be already computed. 
        self.fetch = fetch
        
        
        
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
        
        # Wishlist : Multiple pipelines can be chained together.
        # Answer: Easy! Train a stack of rbms and create an Extractor
        # that takes an arbitrary number of pipelines. It calls activate()
        # on its input data, and passes that down the line.
        
    
    
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
