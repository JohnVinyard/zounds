from __future__ import division
import unittest
from math import ceil

import numpy as np

from zounds.analyze.audiostream import AudioStream
from zounds.nputil import flatten2d
from zounds.testhelper import \
    make_sndfile,filename,remove,RootExtractor,SumExtractor

## AudioStreamTests ##########################################################
class AudioStreamTests(unittest.TestCase):
    
    
    def setUp(self):
        self._to_remove = []
    
    def tearDown(self):
        for tr in self._to_remove:
            remove(tr)
    
    def make_sndfile(self,length,winsize,samplerate,channels = 1):
        fn = make_sndfile(length,winsize,samplerate,channels = channels)
        self._to_remove.append(fn)
        return fn
    
    def get_frames(self,
                   length,
                   step = 1024,
                   ws = 2048,
                   samplerate = 44100,
                   channels = 1):
        
        fn = self.make_sndfile(length, ws, samplerate, channels)
        self._to_remove.append(fn)
        a = AudioStream(fn,samplerate,ws,step)
        l = np.concatenate([w for w in a])
        # If the dtype is object, this probably means that the output from
        # AudioStream was jagged, i.e., not all the windows were the same length
        self.assertNotEqual(object,l.dtype)
        l = flatten2d(l)
        b = ceil((max(0,length - ws) / step) + 1)
        self.assertEqual(b,l.shape[0])
        
    def test_nonexistentfile(self):
        fn = filename()
        self.assertRaises(IOError,lambda : AudioStream(fn).__iter__().next())
        
    def test_lt_windowsize(self):
        self.get_frames(2000)
        
    def test_lt_stepsize(self):
        self.get_frames(1000)
    
    def test_equal_windowsize(self):
        self.get_frames(2048)
    
    def test_two_frames(self):
        self.get_frames(2050)

    # TODO: Switch to unittest2, so this test can be ignored    
    #def test_three_frames(self):
    #    self.get_frames(4096)
    
    def test_four_frames(self):
        self.get_frames(4097)
    
    def test_resample(self):
        # ensure that two sounds with the same length in seconds, but differing
        # samplerates end up with the same number of windows, if the same
        # sampling rate is passed to both audio streams
        seconds = 5
        sr1 = 44100
        sr2 = 48e3
        s1 = sr1 * seconds
        s2 = sr2 * seconds
        ws = 2048
        ss = 1024
        fn1 = self.make_sndfile(s1, ws, sr1, 1)
        fn2 = self.make_sndfile(s2, ws, sr2, 1)
        # note that both audio streams are getting the same sample rate
        # parameter, since we're testing resampling
        as1 = AudioStream(fn1,sr1,ws,ss)
        as2 = AudioStream(fn2,sr1,ws,ss)
        f1 = np.concatenate([c for c in as1])
        f2 = np.concatenate([c for c in as2])
        f1 = flatten2d(f1)
        f2 = flatten2d(f2)
        self.assertEqual(f1.shape[0],f2.shape[0])

## ExtractorTests #############################################################
from zounds.analyze.extractor import \
    Extractor,SingleInput,ExtractorChain,RootlessExtractorChainException



    
    
class NoOpExtractor(Extractor):
    
    def __init__(self,needs, key = None):
        Extractor.__init__(self,needs, key = key)
    
    def dim(self,env):
        raise NotImplemented()
    
    @property
    def dtype(self):
        raise NotImplemented()
        
    def _process(self):
        return self.input[self.sources[0]]

class ShimExtractor(Extractor):
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        Extractor.__init__(self,needs,nframes,step,key)
    
    def dim(self,env):
        raise NotImplemented()
    
    @property
    def dtype(self):
        raise NotImplemented()
    
    def _process(self):
        raise NotImplemented()

class ExtractorTests(unittest.TestCase):
    
    def test_bad_frames_count(self):
        self.assertRaises(ValueError, lambda : ShimExtractor(nframes=-1))
        
    def test_bad_step_size(self):
        self.assertRaises(ValueError,lambda : ShimExtractor(step=0))
        
    def test_is_root(self):
        self.assertTrue(ShimExtractor().is_root)
        
    def test_is_not_root(self):
        re = ShimExtractor()
        se = ShimExtractor(needs=re)
        self.assertFalse(se.is_root)
        
    def test_directly_depends_on(self):
        re = ShimExtractor()
        se = ShimExtractor(needs = re)
        self.assertTrue(se.depends_on(re))
        
    def test_indirectly_depends_on(self):
        re = ShimExtractor()
        se1 = ShimExtractor(needs = re)
        se2 = ShimExtractor(needs = se1)
        self.assertTrue(se2.depends_on(se1))
        self.assertTrue(se2.depends_on(re))
    
    def test_does_not_depend_on(self):
        re = ShimExtractor()
        se1 = ShimExtractor(needs = re)
        se2 = ShimExtractor(needs = re)
        self.assertFalse(se1.depends_on(se2))
        self.assertFalse(se2.depends_on(se1))
        
    def test_depends_on_multi_dependency(self):
        re = ShimExtractor()
        se1 = ShimExtractor(needs = [re])
        se2 = ShimExtractor(needs = [re,se1])
        self.assertTrue(se2.depends_on(re))
        self.assertTrue(se2.depends_on(se1))
    
    def test_equality_different_classes(self):
        re = RootExtractor()
        se = SumExtractor(re,1,1)
        self.assertNotEqual(re,se)
        
    def test_equality_same_class_different_sources(self):
        re = RootExtractor()
        se = SumExtractor(None,1,1)
        d1 = SumExtractor(se,1,1)
        d2 = SumExtractor(re,1,1)
        self.assertNotEqual(d1,d2)
    
    def test_equality_different_nframes(self):
        se1 = SumExtractor(None,1,1)
        se2 = SumExtractor(None,2,1)
        self.assertNotEqual(se1,se2)
    
    def test_equality_different_step(self):
        se1 = SumExtractor(None,1,1)
        se2 = SumExtractor(None,1,2)
        self.assertNotEqual(se1,se2)
    
    def test_equality_equal(self):
        re = RootExtractor()
        se1 = SumExtractor(re,1,1)
        se2 = SumExtractor(re,1,1)
        self.assertEqual(se1,se2)
    
    def test_nframes_abs_both_one(self):
        re = RootExtractor()
        se = SumExtractor(re,1,1)
        self.assertEqual(1,re.nframes_abs())
        self.assertEqual(1,se.nframes_abs())
        
    def test_nframes_abs_two(self):
        re = RootExtractor()
        se1 = SumExtractor(re,2,1)
        se2 = SumExtractor(se1,1,1)
        self.assertEqual(1,re.nframes_abs())
        self.assertEqual(2,se1.nframes_abs())
        self.assertEqual(2,se2.nframes_abs())
    
    def test_nframes_compound(self):
        re = RootExtractor()
        se1 = SumExtractor(re,2,1)
        se2 = SumExtractor(se1,2,1)
        self.assertEqual(1,re.nframes_abs())
        self.assertEqual(2,se1.nframes_abs())
        self.assertEqual(4,se2.nframes_abs())
    
    def test_nframes_multisource(self):
        re = RootExtractor()
        se1 = SumExtractor(re,2,1)
        se2 = SumExtractor(re,1,1)
        se3 = SumExtractor([se1,se2],1,1)
        self.assertEqual(2,se3.nframes_abs())
    
    def test_frames_abs_multi_source_differing_relative_nframes(self):
        '''
            1
           / \
          60 60
          |  |
          \  1
           \/
           1
           
        My previous nframes_abs algorithm reported the absolute nframes value
        of the termintating node to be 3600, because ancestor nodes were
        arranged like so:
        1 x 60 x 60 x 1
        '''
        re = RootExtractor()
        se1 = SumExtractor(re,60,1)
        se2 = SumExtractor(re,60,1)
        se3 = SumExtractor(se2,1,1)
        se4 = SumExtractor([se1,se3],1,1)
        self.assertEqual(60,se4.nframes_abs())
        
    
    def test_step_abs_both_one(self):
        re = RootExtractor()
        se = SumExtractor(re,1,1)
        self.assertEqual(1,re.step_abs())
        self.assertEqual(1,se.step_abs())
    
    def test_step_abs_two(self):
        re = RootExtractor()
        se1 = SumExtractor(re,1,2)
        se2 = SumExtractor(se1,1,1)
        self.assertEqual(1,re.step_abs())
        self.assertEqual(2,se1.step_abs())
        self.assertEqual(2,se2.step_abs())
    
    def test_step_compound(self):
        re = RootExtractor()
        se1 = SumExtractor(re,1,2)
        se2 = SumExtractor(se1,1,2)
        self.assertEqual(1,re.step_abs())
        self.assertEqual(2,se1.step_abs())
        self.assertEqual(4,se2.step_abs())
    
    def test_step_multisource(self):
        re = RootExtractor()
        se1 = SumExtractor(re,1,2)
        se2 = SumExtractor(re,1,1)
        se3 = SumExtractor([se1,se2],1,1)
        self.assertEqual(2,se3.step_abs())
    
    def test_step_abs_multi_source_differing_relative_nframes(self):
        '''
            1
           / \
          60 60
          |  |
          \  1
           \/
           1
           
        My previous step_abs algorithm reported the absolute nframes value
        of the termintating node to be 3600, because ancestor nodes were
        arranged like so:
        1 x 60 x 60 x 1
        '''
        re = RootExtractor()
        se1 = SumExtractor(re,1,60)
        se2 = SumExtractor(re,1,60)
        se3 = SumExtractor(se2,1,1)
        se4 = SumExtractor([se1,se3],1,1)
        self.assertEqual(60,se4.step_abs())
    
    
class SingleInputTests(unittest.TestCase):
    
    def test_root(self):
        self.assertRaises(ValueError, lambda : SingleInput(None))
        
    def test_in_data_singledim(self):
        re = RootExtractor()
        si = SingleInput(re)
        re.collect()
        re.process()
        si.collect()
        data = si.in_data
        self.assertTrue(data is not None)
        expected = np.ones(re.chunksize)
        self.assertTrue(np.all(expected == data))
    
    def test_in_data_multidim(self):
        re = RootExtractor(shape=2)
        si = SingleInput(re)
        re.collect()
        re.process()
        si.collect()
        data = np.array(si.in_data)
        self.assertTrue(data is not None)
        compare = np.ones((1,re.chunksize,re.shape))
        self.assertEqual(compare.shape,data.shape)
        self.assertTrue(np.all(data == compare))
        

## ExtractorChainTests ########################################################
        
class ExtractorChainTests(unittest.TestCase):
    
        
    def test_sort(self):
        re = RootExtractor()
        se = SumExtractor(needs=re,nframes=5,step=1)
        se2 = SumExtractor(needs=se,nframes=1,step=1)
        ec = ExtractorChain([se,se2,re])
        self.assertEqual(re,ec.chain[0])
        self.assertEqual(se,ec.chain[1])
        self.assertEqual(se2,ec.chain[2])
    
    def test_sort_multiple_dependencies(self):
        re = RootExtractor()
        se1 = SumExtractor(needs=re,nframes=1,step=1)
        se2 = SumExtractor(needs=se1,nframes=2,step=1)
        se3 = SumExtractor(needs=[re,se1],nframes=3,step=1)
        ec = ExtractorChain([se3,re,se1,se2])
        self.assertEqual(re,ec.chain[0])
        self.assertEqual(se1,ec.chain[1])
        self.assertTrue(se2 in ec.chain[2:])
        self.assertTrue(se3 in ec.chain[2:])
    
    def test_empty_extractor_chain(self):
        self.assertRaises(ValueError,lambda : ExtractorChain([]))
        
    def test_single_extractor_chain(self):
        re = RootExtractor()
        ec = ExtractorChain(re)
        d = ec.collect()
        self.assertEqual(1,len(d))
        self.assertTrue(d.has_key(re))
        v = np.concatenate(d[re])
        self.assertEqual(10,v.size)
        self.assertTrue(all([q == 1 for q in v]))
    
    def test_two_extractor_chain_no_step(self):
        re = RootExtractor()
        se = SumExtractor(needs=re,nframes=2,step=1)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        self.assertEqual(2,len(d))
        self.assertTrue(d.has_key(re))
        self.assertTrue(d.has_key(se))
        rev = np.concatenate(d[re])
        sev = np.concatenate(d[se])
        self.assertEqual(10,len(rev))
        self.assertEqual(10,len(sev))
        self.assertTrue(all([q == 1 for q in rev]))
        self.assertTrue(all([q in (1,2) for q in sev]))
        
    def test_two_extractor_chain_twodim(self):
        re = RootExtractor(shape=10)
        se = SumExtractor(needs=re,nframes=2,step=2)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        self.assertTrue(2,len(d))
        self.assertTrue(d.has_key(re))
        self.assertTrue(d.has_key(se))
        rev = np.array(d[re])
        sev = np.array(d[se])
        self.assertEqual((2,5,10),rev.shape)
        # sev is jagged here, and consequently has a dtype of object.  We're
        # calling concatenate so we can easily find out how many elements sev
        # contains
        sev = np.concatenate(sev.tolist())
        self.assertEqual(5,sev.size)
        self.assertTrue(all([s==20 for s in sev]))
        
    def test_no_root(self):
        re = RootExtractor()
        se = SumExtractor(needs=re,nframes=1,step=1)
        self.assertRaises(RootlessExtractorChainException,
                          lambda : ExtractorChain([se]))
        
    def test_extractor_chain_with_multi_dependency_extractor(self):
        re = RootExtractor()
        se1 = SumExtractor(needs = re, nframes = 1, step = 1)
        se2 = SumExtractor(needs = [re,se1], nframes = 1, step = 1)
        ec = ExtractorChain([se1,re,se2])
        d = ec.collect()
        self.assertTrue(3,len(d))
        self.assertTrue(d.has_key(re))
        self.assertTrue(d.has_key(se1))
        self.assertTrue(d.has_key(se2))
        sev = np.concatenate(d[se2])
        self.assertEqual((10,),sev.shape)
        self.assertTrue(all([s == 2 for s in sev]))
        
    def test_extractor_multi_resolution(self):
        re = RootExtractor()
        se1 = SumExtractor(needs = re, nframes = 2, step = 1)
        se2 = SumExtractor(needs = se1, nframes = 2, step = 1)
        ec = ExtractorChain([se2,re,se1])
        d = ec.collect()
        self.assertTrue(4,len(d))
        self.assertTrue(d.has_key(re))
        self.assertTrue(d.has_key(se1))
        self.assertTrue(d.has_key(se2))
        rev = np.concatenate(d[re])
        se1v = np.concatenate(d[se1])
        se2v = np.concatenate(d[se2])
        self.assertEqual(10,len(rev))
        self.assertEqual(10,len(se1v))
        self.assertEqual(10,len(se2v))
        self.assertTrue(all([s in (1,2) for s in se1v]))
        self.assertTrue(all([s in (3,4,1) for s in se2v]))
    
    def test_noop_singledim(self):
        re = RootExtractor()
        se = NoOpExtractor(needs = re)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        inp = np.array(d[re])
        self.assertEqual((2,5),inp.shape)
        output = np.array(d[se])
        self.assertEqual((2,5),output.shape)
        
    def test_noop_multidim(self):
        re = RootExtractor(shape=10)
        se = NoOpExtractor(needs = re)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        inp = np.array(d[re])
        self.assertEqual((2,5,10),inp.shape)
        output = np.array(d[se])
        self.assertEqual((2,5,10),output.shape)
    
    def test_getitem_int(self):
        re = ShimExtractor(key = 'oh')
        se = ShimExtractor(key = 'hai', needs = re)
        ec = ExtractorChain([se,re])
        
        e = ec[0]
        self.assertTrue(e is re)
    
    def test_getitem_int_index_error(self):
        re = ShimExtractor(key = 'oh')
        se = ShimExtractor(key = 'hai', needs = re)
        ec = ExtractorChain([se,re])
        
        self.assertRaises(IndexError,lambda : ec[3]) 
        
    def test_getitem_string_key_missing(self):
        re = ShimExtractor(key = 'oh')
        se = ShimExtractor(key = 'hai', needs = re)
        ec = ExtractorChain([se,re])
        
        self.assertRaises(KeyError, lambda : ec['chzburger'])
    
    def test_getitem_string_key_present(self):
        re = ShimExtractor(key = 'oh')
        se = ShimExtractor(key = 'hai', needs = re)
        ec = ExtractorChain([se,re])
        
        self.assertTrue(ec['hai'] is se)
    
    def test_getitem_invalid_key_type(self):
        re = ShimExtractor(key = 'oh')
        se = ShimExtractor(key = 'hai', needs = re)
        ec = ExtractorChain([se,re])
        
        self.assertRaises(ValueError, lambda : ec[10:20])
    
    def test_prune(self):
        re = ShimExtractor(key = 'oh')
        e1 = ShimExtractor(key = 'hai', needs = re)
        e2 = ShimExtractor(key = 'chzburger', needs = re)
        
        
        ec = ExtractorChain([e1,e2,re]).prune('chzburger')
        self.assertEqual(2,len(ec))
        self.assertTrue(e2 in ec)
        self.assertFalse(e1 in ec)
    
    def test_prune_multiple_keys(self):
        re = ShimExtractor(key = 'oh')
        e1 = ShimExtractor(key = 'hai', needs = re)
        e2 = ShimExtractor(key = 'chzburger', needs = e1)
        e3 = ShimExtractor(key = 'canhas', needs = re)
        e4 = ShimExtractor(key = 'in yur computer', needs = re)
        
        ec = ExtractorChain([re,e1,e2,e3,e4]).prune('chzburger','canhas')
        self.assertEqual(4,len(ec))
        self.assertFalse(e4 in ec)
        
    
