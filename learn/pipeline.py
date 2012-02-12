class Pipeline(object):
    '''
    A pipeline knows how to
        - fetch training samples
        - preprocess them
        - train a learning machine
        - save preprocessing and learning machine parameters
        - reconstruct this pipeline in the future
        
    This implies
        - generalized fetchers
        - preprocessors than can save state
        - learning machines that can save state
        
    Checkpoints during training would also be nice
    
    What about a chain of pipelines, for training
    multiple layers of an nnet in row?
    '''
    
    def __init__(self,datadir):
        object.__init__(self)
        
        # something that knows how to fetch training examples
        fetch = None
        
        # something that knows how to preprocess data 
        preprocess = None
        
        # something that knows how to train on data, and can
        # describe future data based on that training.
        learner = None
        
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