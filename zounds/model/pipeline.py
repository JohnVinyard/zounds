from datetime import datetime
from zounds.nputil import flatten2d
from model import Model
from multiprocessing import Pool
from zounds.util import tostring

class MetaPipeline(type):
    
    def __init__(self,name,bases,attrs):
        super(MetaPipeline,self).__init__(name,bases,attrs)
    
    def __getitem__(self,key):
        return self.controller()[key]
    
    def __delitem__(self,key):
        del self.controller()[key]


class Pipeline(Model):
    '''
    A Pipeline defines a workflow consisting of three parts:
    
        * A strategy for fetching training data.  This step is only used prior \
          to training.
        * A strategy for preprocessing data.  This step is important both before \
          and after training.  Once the learning algorithm has been trained, and \
          is being used as a feature extractor, all incoming data must be preprocessed \
          precisely as it was prior to training
        * A learning algorithm, which transforms data into some more useful \
          representation
    
    Pipelines are persisted to a data store once trained, and can be retrieved \
    by the id they were assigned at creation::
    
        >>> p = Pipeline['pipeline/audio_256']
        Pipeline(
            preprocess = UnitNorm(),
            learn = KMeans(n_centroids = 500),
            trained_date = 2012-09-19 16:12:52.578097,
            id = pipeline/audio_256,
            fetch = PrecomputedFeature(feature = audio, nframes = 1))
    '''
    
    __metaclass__ = MetaPipeline
    
    def __init__(self,_id,fetch,preprocess,learn):
        '''__init__
        
        :param _id: a filepath-like identifier for this pipeline, .e.g \
        :code:`'pipeline/bark/kmeans_500'`.  This _id can be used to retrieve the Pipeline \
        from the data store, once trained.
        
        :param fetch: A :py:class:`~zounds.learn.fetch.Fetch`-derived class, responsible \
        for gathering training data from the data store
        
        :param preprocess: A :py:class:`~zounds.learn.preprocess.Preprocess`-derived \
        class, responsible for preparing the data for training and/or feature \
        extraction
        
        :param learn: A :py:class:`~zounds.learn.learn.Learn`-derived class, \
        responsible for learning a new representation of the input data.
        '''
        Model.__init__(self)        
        self._id = _id
        self.fetch = fetch
        self.preprocess = preprocess
        self.learn = learn
        # the date this pipeline completed training
        self.trained_date = None
    
    def __repr__(self):
        return tostring(self,short = False,id = self._id, fetch = self.fetch, 
                        preprocess = self.preprocess,learn = self.learn,
                        trained_date = self.trained_date)
    
    def __str__(self):
        return tostring(self,id = self._id, fetch = self.fetch, 
                        preprocess = self.preprocess, learn = self.learn)
    
    @property
    def dim(self):
        try:
            return self.learn.dim
        except NameError:
            raise NotImplemented('%s has not implemented a dim property' % \
                                self.learn.__class__.__name__)
    
    def train(self,nexamples,stopping_condition,data = None):
        '''train
        
        Fetch data, preprocess it, and pass it along to a learning algorithm.
        
        :param nexamples: Passed along to :code:`self.fetch.__call__`, this will \
        determine the number of training examples that are fetched from the data \
        store
        
        :param stopping_condition: A callable, which is specific to the \
        :py:class:`~zounds.learn.learn.Learn`-derived class.  It is evaluated \
        periodically by :py:meth:`~zounds.learn.learn.Learn.train` to decide \
        when learning is complete.
        '''
        if self.controller().id_exists(self._id):
            raise ValueError(\
                    '_id %s already exists. Please delete it before proceeding'\
                     % self._id)
        data = self.fetch(nexamples = nexamples) if data is None else data
        # KLUDGE: Is this always OK?
        if data is not None:
            data = flatten2d(data)
        data = self.preprocess(data)
        # TODO checkpoints, incremental save
        self.learn.train(data,stopping_condition)
        self.trained_date = datetime.utcnow()
        self.store()
    
    def __call__(self,data):
        '''__call__
        
        Activate the Pipeline on an arbitrary number of input data. Data will \
        be preprocessed and passed along to the learning algorithm for \ 
        transformation.
        
        :param data: A numpy array containing data to be transformed.  The data \
        must be the same size as the data that this Pipeline trained on, \
        in all dimensions except for the first.  E.g., if trained on a feature \
        of dimension 10, the second dimension of data must always be 10, but the \
        first dimension can be anything.
        '''
        data = self.preprocess(data)
        return self.learn(data)
    
    def store(self):
        self.controller().store(self)


def _train(args):
    pipeline,nexamples,data,stop_cond = args
    pipeline.train(nexamples,stop_cond, data = data)

def train_many(pipelines,nsamples,stop_condition,nprocesses = 4):
    pool = Pool(nprocesses)
    for i in range(0,len(pipelines),nprocesses):
        args = [(p,nsamples,p.fetch(nsamples),stop_condition) \
                    for p in pipelines[i : i + nprocesses]]
        pool.map(_train,args)
     
