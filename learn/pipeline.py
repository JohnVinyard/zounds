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
        
        # the path to a directory where
        # all data for this pipeline will be
        # stored
        self.datadir = datadir