import unittest
from frame import Frames,Feature
from environment import Environment
from analyze.feature import RawAudio,FFT,Loudness

class FrameModelTests(unittest.TestCase):
    
    
    def setUp(self):
        self.source = 'test'
        self.orig_env = Environment.instance
    
    def tearDown(self):
        Environment.instance = self.orig_env
    
    # FramesModel
    
    def test_no_features(self):
        self.fail()
    
    def test_extractor_chain(self):
        self.fail()
        
    def test_create_db(self):
        self.fail()
    
    def test_new_feature_db_in_sync(self):
        self.fail()
        
    def test_unchanged_db_in_sync(self):
        self.fail()
    
    def test_add_feature_stored(self):
        self.fail()
    
    def test_add_feature_not_stored(self):
        self.fail()
    
    def test_remove_feature_stored(self):
        self.fail()
    
    def test_remove_feature_not_stored(self):
        self.fail()
        
    def test_add_feature_depency_already_computed(self):
        self.fail()
    
    def test_add_feature_dependency_not_stored(self):
        self.fail()
    
    def test_same_num_features_dependency_changed(self):
        self.fail()
    
    def test_force_update(self):
        self.fail()
    
    def test_force_no_sync(self):
        # e.g., an extractor class' name gets changed, but nothing else about
        # it does
        self.fail()
        
    def test_removed_feature_upon_which_other_feature_depends(self):
        self.fail()
        
        