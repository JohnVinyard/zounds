import unittest

import numpy as np


from frame import Frames,Feature,Precomputed
from pattern import FilePattern
from environment import Environment
from analyze.feature.spectral import FFT,Loudness,SpectralCentroid
from data.frame import DictFrameController



    
class FrameModelTests(unittest.TestCase):
    
    
    def setUp(self):
        Environment._test = True
        self.orig_env = Environment.instance
        
    
    def tearDown(self):
        Environment._test = False
        Environment.instance = self.orig_env
    
    def mock_pattern(self):
        return FilePattern('0','0','0','/some/file.wav')
    
    def test_dimensions_correct_keys(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
            
        Environment('test',
                    FM1,
                    DictFrameController,
                    (FM1,),
                    {})
        
        dims = FM1.dimensions()
        self.assertEqual(7,len(dims))
        self.assertTrue(dims.has_key('fft'))
        self.assertTrue(dims.has_key('loudness'))
    
    def test_dimensions_correct_values(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        class AudioConfig:
            samplerate = 44100
            windowsize = 4096
            stepsize = 1024
            
        Environment('test',
                    FM1,
                    DictFrameController,
                    (FM1,),
                    {},
                    AudioConfig)
        
        # KLUDGE: I should really write my own Extractor-derived class for this
        # test, but the FFT is such a foundational feature that it's unlikely
        # to go away or change its behavior. The real purpose of this test
        # is to demonstrate that the FrameModel can construct features with
        # the correct dimensions, given its context
        dims = FM1.dimensions()
        self.assertEqual(3,len(dims['fft']))
        
        t = dims['fft']
        self.assertEqual(t[0],2048)
        
        
    def test_equality(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        self.assertEqual(FM1.fft,FM2.fft)
    
    def test_unchanged(self):
        
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertFalse(update)
        self.assertFalse(delete)
        
        chain = FM2.extractor_chain(self.mock_pattern(), 
                                    transitional = True,
                                    recompute = recompute)
        # lop of the first item, because the MetaData extractor is never
        # precomputed
        self.assertTrue(all([isinstance(e,Precomputed) for e in chain.chain[1:]]))
        
    def test_unstore(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
        
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertFalse(update)
        self.assertTrue(delete)
        self.assertEqual(1,len(delete))
        self.assertTrue(delete.has_key('loudness'))
        
    
    def test_unstored_to_stored(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        self.assertTrue(add)
        self.assertFalse(update)
        self.assertFalse(delete)
        self.assertEqual(1,len(add))
        self.assertEqual(FM2.loudness,add['loudness'])
    
    def test_add(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        self.assertTrue(add)
        self.assertFalse(update)
        self.assertFalse(delete)
        self.assertEqual(1,len(add))
        self.assertEqual(FM2.loudness,add['loudness'])
    
    def test_update(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft,nframes=1)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft,nframes=2)
        
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertTrue(update)
        self.assertFalse(delete)
        self.assertEqual(1,len(update))
        self.assertEqual(FM2.loudness,update['loudness'])
        
        chain = FM2.extractor_chain(self.mock_pattern(), 
                                    transitional = True, 
                                    recompute = recompute)
        
        precomputed = filter(lambda e : isinstance(e,Precomputed), chain.chain)
        self.assertEqual(6,len(precomputed))    
    
    def test_delete(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            
        
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertFalse(update)
        self.assertTrue(delete)
        self.assertEqual(1,len(delete))
        self.assertTrue(delete.has_key('loudness'))
        
        chain = FM2.extractor_chain(self.mock_pattern(), 
                                    transitional = True,
                                    recompute = recompute)
        
        precomputed = filter(lambda e : isinstance(e,Precomputed), chain.chain)
        self.assertEqual(6,len(precomputed))
        
        
    def test_update_lineage(self):
        
        class FM1(Frames):
            l1 = Feature(Loudness,store=True,nframes=1)
            l2 = Feature(Loudness,store=True,nframes=4,needs=l1)
            
        class FM2(Frames):
            l1 = Feature(Loudness,store=True,nframes=2)
            l2 = Feature(Loudness,store=True,nframes=4,needs=l1)
            
        Environment('test',
                    FM2,
                    DictFrameController,
                    (FM2,),
                    {})
        
        add,update,delete,recompute = FM1.update_report(FM2)
        
        self.assertFalse(add)
        self.assertTrue(update)
        self.assertFalse(delete)
        self.assertEqual(2,len(update))
        self.assertEqual(FM2.l1,update['l1'])
        self.assertEqual(FM2.l2,update['l2'])
        
        chain = FM2.extractor_chain(self.mock_pattern(), 
                                    transitional = True, 
                                    recompute = recompute)
        
        precomputed = filter(lambda e : isinstance(e,Precomputed), chain.chain)
        self.assertEqual(5,len(precomputed))
        
    
     
    def test_get(self):
        
        class FM(Frames):
            fft = Feature(FFT,store = True, needs = None)
            loudness = Feature(Loudness, store = True, need = fft)
            
        class Controller(DictFrameController):
            
            def get(self,address):
                return {'_id' : 1,
                        'audio' : 1,
                        'source' : 'test',
                        'external_id' : '0',
                        'framen' : 0,
                        'fft' : 1, 
                        'loudness' : 2}
            
        Environment('test',
                    FM,
                    Controller,
                    (FM,),
                    {})
        frames = FM['some_id']
        self.assertEqual(1,frames.fft)
        self.assertEqual(2,frames.loudness)
        
    def test_get_unstored_feature(self):
        '''
        Features that aren't stored should be computed on the fly when
        their attribute name is accessed
        '''
        class FM(Frames):
            fft = Feature(FFT,store = False, needs = None)
            loudness = Feature(Loudness, store = True, needs = fft)
            
        class Controller(DictFrameController):
            
            def get(self,address):
                return {'_id' : 1,
                        'audio' : np.ones(1000),
                        'source' : 'test',
                        'external_id' : '0',
                        'framen' : 0,
                        'fft' : 1, 
                        'loudness' : 2}
            
        Environment('test',
                    FM,
                    Controller,
                    (FM,),
                    {})
        frames = FM['some_id']
        self.assertTrue(isinstance(frames.fft,np.ndarray))
        self.assertTrue(len(frames.fft) > 0)
    
    
    def test_get_no_results(self):
        class FM(Frames):
            fft = Feature(FFT,store = False, needs = None)
            loudness = Feature(Loudness, store = True, needs = fft)
            
        class Controller(DictFrameController):
            
            def get(self,address):
                return []
            
        Environment('test',
                    FM,
                    Controller,
                    (FM,),
                    {})
        
        self.assertRaises(KeyError, lambda : FM['some_id'])
    
    def mock_frames_instance(self,l):
        class FM(Frames):
            fft = Feature(FFT,store = False, needs = None)
            loudness = Feature(Loudness, store = True, need = fft)
            
        class Controller(DictFrameController):
            
            def get(self,address):
                r = np.recarray(l,dtype=[('_id','a10'),
                                          ('audio',np.float32),
                                          ('source','a10'),
                                          ('external_id','a10'),
                                          ('framen',np.int32),
                                          ('fft',np.float32),
                                          ('loudness',np.float32)])
                r[:] = l
                return r
            
        Environment('test',
                    FM,
                    Controller,
                    (FM,),
                    {})
        return FM,FM['some_id'] 
        
    
    def test_instance_len(self):
        cls,frames = self.mock_frames_instance(10)
        self.assertEqual(10,len(frames))
        
    
    def test_instance_getitem_str_exists(self):
        cls,instance = self.mock_frames_instance(10)
        f = instance['loudness']
        self.assertTrue(isinstance(f,np.ndarray))
        self.assertEqual(10,len(f))
    
    def test_instance_getitem_str_not_exists(self):
        cls,instance = self.mock_frames_instance(10)
        self.assertRaises(KeyError,lambda : instance['centroid'])
    
    def test_instance_getitem_str_unstored(self):
        self.fail('__getitem__ does not work for non-stored features yet')
    
    def test_instance_getitem_feature_exists(self):
        cls,instance = self.mock_frames_instance(10)
        f = instance[cls.loudness]
        self.assertTrue(isinstance(f,np.ndarray))
        self.assertEqual(10,len(f))
    
    def test_instance_getitem_feature_not_exists(self):
        feature = Feature(SpectralCentroid,store = True, needs = None)
        cls,instance = self.mock_frames_instance(10)
        self.assertRaises(KeyError, lambda : instance[feature] )
    
    def test_instance_getitem_feature_unstored(self):
        self.fail('__getitem__ does not work for non-stored features yet')
        
    
    def test_instance_getitem_slice(self):
        self.fail()
    
    def test_instance_getitem_int(self):
        self.fail()
        
    def test_instance_getitem_wrong_type(self):
        cls,instance = self.mock_frames_instance(10)
        self.assertRaises(ValueError,lambda : instance[object()])
    
    