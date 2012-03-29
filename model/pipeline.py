from datetime import datetime

from model import Model

class MetaPipeline(type):
    
    def __init__(self,name,bases,attrs):
        super(MetaPipeline,self).__init__(name,bases,attrs)
    
    def __getitem__(self,key):
        return self.controller()[key]
    
    def __delitem__(self,key):
        del self.controller()[key]

# TODO: Define __getitem__ on class
class Pipeline(Model):
    '''
    A generic pipeline that can be used to chain together methods for:
        - fetching data
        - preprocessing data
        - running supervised or unsupervised algorithms on the data
    
    A trained pipeline can be treated as a feature extractor.
    '''
    
    __metaclass__ = MetaPipeline
    
    def __init__(self,_id,fetch,preprocess,learn):
        '''
        param _id: a filepath-like identifier for this specific pipeline, .e.g
        pipeline/bark/rbm/rbm2000
        param fetch: a learn.fetch.Fetch implementation
        param preprocess: a learn.preprocess.Preprocess implementation
        param learn: a learn.learn.Learn implementation
        '''
        Model.__init__(self)        
        self._id = _id
        self.fetch = fetch
        self.preprocess = preprocess
        self.learn = learn
        # the date this pipeline completed training
        self.trained_date = None
        

    def train(self,stopping_condition):
        '''
        param stopping_condition: A callable that is specific to the 
        Learn-derived class. It is evaluated periodically by Learn.train to 
        decide when learning is complete.
        '''
        data = self.fetch()
        data = self.preprocess(data)
        # TODO checkpoints, incremental save
        self.learn.train(data,stopping_condition)
        self.trained_date = datetime.utcnow()
        self.store()
    
    def __call__(self,data):
        data = self.preprocess(data)
        return self.learn(data)
    
    def store(self):
        self.controller().store(self)
        
