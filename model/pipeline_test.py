import unittest

import numpy as np

from environment import Environment
from frame import Frames
from data.frame import DictFrameController
from data.pipeline import DictPipelineController
from pipeline import Pipeline
from learn.fetch import Fetch
from learn.preprocess import Preprocess
from learn.learn import Learn

class MockFetch(Fetch):
    
    def __init__(self,shape,value):
        self.shape = shape
        self.value = value
        Fetch.__init__(self)
        
    
    def __call__(self,nexamples = None):
        arr = np.ndarray(self.shape)
        arr[:] = self.value
        return arr

class AddPreprocess(Preprocess):
    
    def __init__(self):
        Preprocess.__init__(self)
        self.to_add = None
    
    def _preprocess(self,data):
        if None is self.to_add:
            self.to_add = np.average(data)
        
        data += self.to_add
        return data

class MockLearn(Learn):
    
    def __init__(self):
        Learn.__init__(self)
    
    def train(self,data,stopping_condition):
        pass
    
    def __call__(self,data):
        return data * 2
    

class PipelineTests(unittest.TestCase):
    
    def setUp(self):
        Environment._test = True
        class FM(Frames):
            pass
        self.env = Environment('test',
                               FM,
                               DictFrameController,
                               (FM,),
                               {Pipeline : DictPipelineController()})
        
        
    
    def tearDown(self):
        Environment._test = False
    
    def test_train_call(self):
        p = Pipeline('test/test',
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        newdata = np.ndarray((11,2))
        newdata[:] = 3
        
        features = p(newdata)
        # We expect that the average of the original training data will have
        # been added to the new data, and then multiplied by 2
        self.assertTrue(np.all(8 == features))
    
    def test_train_store_call(self):
        key = 'test/test'
        p = Pipeline(key,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        newdata = np.ndarray((11,2))
        newdata[:] = 3
        
        p2 = Pipeline[key]
        features = p2(newdata)
        # We expect that the average of the original training data will have
        # been added to the new data, and then multiplied by 2
        self.assertTrue(np.all(8 == features))
    
    def test_train_store_date(self):
        key = 'test/test'
        p = Pipeline(key,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        newdata = np.ndarray((11,2))
        newdata[:] = 3
        
        p2 = Pipeline[key]
        self.assertFalse(p2.trained_date is None)
    
    def test_key_error(self):
        self.assertRaises(KeyError, lambda : Pipeline['somekey'])

    