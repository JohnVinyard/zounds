import unittest
import os.path
import shutil
from uuid import uuid4

from zounds.environment import Environment
from pipeline import PickledPipelineController
from frame import DictFrameController

from zounds.model.pipeline import Pipeline
from zounds.model.frame import Frames
 
from zounds.model.test_pipeline import MockFetch,AddPreprocess,MockLearn
from zounds.testhelper import remove

class PickledLearningControllerTests(unittest.TestCase):
    
    def setUp(self):
        Environment._test = True
        self.to_cleanup = []
        class FM(Frames):
            pass
        
        self.env = Environment('test',
                               FM,
                               DictFrameController,
                               (FM,),
                               {Pipeline : PickledPipelineController()})
        self.controller = self.env.data[Pipeline]
    
    def tearDown(self):
        for tc in self.to_cleanup:
            remove(tc)
            
        
    def test_store(self):
        key = uuid4().hex
        fn = self.controller._filename(key)
        p = Pipeline(key,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        self.to_cleanup.append(fn)
        self.assertTrue(os.path.exists(fn))
        
    def test_store_nested_directory(self):
        
        dir = uuid4().hex
        key = uuid4().hex
        fn = self.controller._filename(key)
        realpath = os.path.join(dir,fn)
        wholekey = os.path.join(dir,key)
        p = Pipeline(wholekey,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        self.to_cleanup.append(dir)
        self.assertTrue(os.path.exists(realpath))
        
    
    def test_get_item(self):
        key = uuid4().hex
        fn = self.controller._filename(key)
        p = Pipeline(key,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        self.to_cleanup.append(fn)
        p2 = Pipeline[key]
        self.assertFalse(None is p2)
        self.assertEqual(1,p2.preprocess.to_add)
        self.assertFalse(None is p2.trained_date)
        
    def test_get_item_key_error(self):
        self.assertRaises(KeyError, lambda : Pipeline['somekey'])
    
    def test_delete(self):
        key = uuid4().hex
        fn = self.controller._filename(key)
        p = Pipeline(key,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        p.train(100,lambda : True)
        self.to_cleanup.append(fn)
        self.assertTrue(os.path.exists(fn))
        del Pipeline[key]
        self.assertFalse(os.path.exists(fn))
        self.to_cleanup.remove(fn)
    
    def test_two_pipelines(self):
        k1 = uuid4().hex
        fn1 = self.controller._filename(k1)
        p1 = Pipeline(k1,
                     MockFetch((100,2),1),
                     AddPreprocess(),
                     MockLearn())
        
        k2 = uuid4().hex
        fn2 = self.controller._filename(k2)
        p2 = Pipeline(k2,
                      MockFetch((100,2),2),
                      AddPreprocess(),
                      MockLearn())
        p1.train(100,lambda : True)
        self.to_cleanup.append(fn1)
        p2.train(100,lambda : True)
        self.to_cleanup.append(fn2)
        
        p1r = Pipeline[k1]
        p2r = Pipeline[k2]
        self.assertEqual(1,p1.preprocess.to_add)
        self.assertEqual(2,p2.preprocess.to_add)