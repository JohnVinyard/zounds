from datetime import datetime

from model import Model


class Pipeline(Model):
    '''
    '''
    
    def __init__(self,_id,fetch,preprocess,learn, training_complete_condition):
        '''
        param _id: a filepath-like identifier for this specific pipeline, .e.g
        pipeline/bark/rbm/rbm2000
        param fetch: a learn.fetch.Fetch implementation
        param preprocess: a learn.preprocess.Preprocess implementation
        param learn: a learn.learn.Learn implementation
        param training_complete_condition: a training stopping condition specific
        to the learn implementation
        '''
        Model.__init__(self)        
        self._id = _id
        self.fetch = fetch
        self.preprocess = preprocess
        self.learn = learn
        self.stopping_condition = training_complete_condition
        # the date this pipeline completed training
        self.trained_date = None
        

    def train(self):
        data = self.fetch()
        data = self.preprocess(data)
        self.save()
        # TODO checkpoints, incremental save
        self.learn.train(data,self.stopping_condition)
        self.trained_date = datetime.utcnow()
        self.store()
    
    def __call__(self,data):
        data = self.preprocess(data)
        return self.learn(data)
    
    def store(self):
        self.controller().store(self)
        
