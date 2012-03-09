import unittest
from frame import Frames,Feature,Precomputed
from environment import Environment
from analyze.feature import FFT,Loudness
from data.frame import DictFrameController

class FrameModelTests(unittest.TestCase):
    
    
    def setUp(self):
        Environment._test = True
        self.orig_env = Environment.instance
        
    
    def tearDown(self):
        Environment._test = False
        Environment.instance = self.orig_env
        
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
        
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertFalse(update)
        self.assertFalse(delete)
        self.assertTrue(all([isinstance(e,Precomputed) for e in chain.chain]))
        
    def test_unstore(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
        
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
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
        
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
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
        
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
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
        
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertTrue(update)
        self.assertFalse(delete)
        self.assertEqual(1,len(update))
        self.assertEqual(FM2.loudness,update['loudness'])
        
        precomputed = filter(lambda e : isinstance(e,Precomputed), chain.chain)
        self.assertEqual(2,len(precomputed))    
    
    def test_delete(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            
        
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
        self.assertFalse(add)
        self.assertFalse(update)
        self.assertTrue(delete)
        self.assertEqual(1,len(delete))
        self.assertTrue(delete.has_key('loudness'))
        
        precomputed = filter(lambda e : isinstance(e,Precomputed), chain.chain)
        self.assertEqual(2,len(precomputed))
        
        
    def test_update_lineage(self):
        
        class FM1(Frames):
            l1 = Feature(Loudness,store=True,nframes=1)
            l2 = Feature(Loudness,store=True,nframes=4,needs=l1)
            
        class FM2(Frames):
            l1 = Feature(Loudness,store=True,nframes=2)
            l2 = Feature(Loudness,store=True,nframes=4,needs=l1)
            
        Environment('test',FM2,{FM2 : DictFrameController})
        add,update,delete,chain = FM1.update_report(FM2)
        
        self.assertFalse(add)
        self.assertTrue(update)
        self.assertFalse(delete)
        self.assertEqual(2,len(update))
        self.assertEqual(FM2.l1,update['l1'])
        self.assertEqual(FM2.l2,update['l2'])
        
        precomputed = filter(lambda e : isinstance(e,Precomputed), chain.chain)
        self.assertEqual(1,len(precomputed))
        
    def test_no_filename_or_transition(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        self.assertRaises(ValueError, 
                          lambda : FM1.extractor_chain(filename=None, transitional=False))
        
    def test_transition_before_update_report(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
        
        self.assertRaises(ValueError,
                          lambda : FM1.extractor_chain(transitional=False))
        
        
        
        
    
    