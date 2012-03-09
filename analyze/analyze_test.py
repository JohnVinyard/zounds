from __future__ import division
import numpy as np
import unittest
from analyze.audiostream import \
    AudioStream,BadSampleRateException,BadStepSizeException
from scikits.audiolab import Sndfile,Format
from uuid import uuid4
from os import remove
from math import floor

# TODO: How do I cleanup wave files for failed tests?

 ## AudioStreamTests ##########################################################
class AudioStreamTests(unittest.TestCase):
    
        
    def filename(self):
        return '%s.wav' % str(uuid4())
    
    def make_signal(self,length,winsize):
        signal = np.ndarray(int(length))
        for i,w, in enumerate(xrange(0,int(length),winsize)):
            signal[w:w+winsize] = i
        return signal
    
    def make_sndfile(self,length,winsize,samplerate,channels=1):
        signal = self.make_signal(length, winsize)
        filename = self.filename() 
        sndfile = Sndfile(filename,'w',Format(),channels,samplerate)
        if channels == 2:
            signal = np.tile(signal,(2,1)).T
        sndfile.write_frames(signal)
        sndfile.close()
        return filename
        
    def remove_sndfile(self,filename):
        remove(filename)
    
    def fail(self):
        self.assertTrue(False)
        
    def test_nonexistentfile(self):
        fn = self.filename()
        self.assertRaises(IOError,AudioStream,fn)
        
    def test_badsamplerate(self):
        fn = self.make_sndfile(44101,2048,22050)
        a = AudioStream(fn)
        self.assertRaises(BadSampleRateException, lambda : a.__iter__().next())
        self.remove_sndfile(fn)
        
    def test_badstepsize(self):
        fn = self.make_sndfile(44101,2048,44100)
        self.assertRaises(BadStepSizeException,AudioStream,fn,44100,2048,1023)
        self.remove_sndfile(fn)
        
    def get_frames(self,
                   length,
                   step = 1024,
                   ws = 2048,
                   samplerate = 44100,
                   channels = 1):
        
        fn = self.make_sndfile(length, ws, samplerate, channels)
        a = AudioStream(fn,samplerate,ws,step)
        l = [w for w in a]
        # check that all frames are the proper length
        [self.assertEqual(ws,len(q)) for q in l]
        # get the expected length, in frames
        el = floor(length/step)
        # check that the number of frames is as expected
        self.assertEqual(el,len(l))
        n = (ws/step)
        # do we expect the last frame to be padded?
        padded = (length/step) % 1
        for i,q in enumerate(l):
            qs = set(q)
            if el > 1 and padded and i >= len(l) - (n-1):
                self.assertEqual(3,len(qs))
            elif not i or not i % n:
                self.assertEqual(1,len(qs))
            else:
                self.assertEqual(2,len(qs))
        self.remove_sndfile(fn)
        
    def test_fileshorterthanwindowsize(self):
        self.get_frames(2000)
        
    def test_fileshorterthanstepsize(self):
        self.get_frames(1000)
        
    def test_fileshorterthanchunksize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk-1))
        
    def test_fileevenlydivisiblebychunksize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2))
        
    def test_filenotevenlydivisiblebychunksize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*1.53))
            
    def test_quarterstepsize(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2),step=512)  
    
    def test_quarterstepsize_notevenlydivisible(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*(2.1)),step=512)
        
    def test_stereo(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2),channels=2)
    
    def test_stepsize_equals_windowsize_evenly_divisible(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*2),step=2048)
        
    def test_stepsize_equals_windowsize_not_evenly_divisible(self):
        self.get_frames(2048*(AudioStream._windows_in_chunk*3.97),step=2048)


## ExtractorTests #############################################################
from analyze.extractor import \
    Extractor,SingleInput,ExtractorChain,RootlessExtractorChainException
    
class RootExtractor(Extractor):
    
    def __init__(self,shape=1,totalframes=10):
        self.shape = shape
        Extractor.__init__(self)
        self.framesleft = totalframes
    
    @property
    def dim(self):
        return self.shape
    
    @property
    def dtype(self):
        return np.int32
    
    def _process(self):
        self.framesleft -= 1
        
        # TODO: Derived classes shouldn't have to know to set done to
        # True and return None. There should be a simpler way
        if self.framesleft < 0:
            self.done = True
            self.out = None
            return None
        
        if self.shape == 1:
            return 1
        return np.ones(self.shape)

class SumExtractor(Extractor):
    
    def __init__(self,needs,nframes,step):
        Extractor.__init__(self,needs,nframes,step)
        
    @property
    def dim(self):
        return (1,)
    
    @property
    def dtype(self):
        return np.int32
        
    def _process(self):
        return np.sum([v for v in self.input.values()]) 
    
    
class NoOpExtractor(Extractor):
    
    def __init__(self,needs):
        Extractor.__init__(self,needs)
    
    @property
    def dim(self):
        raise NotImplemented()
    
    @property
    def dtype(self):
        raise NotImplemented()
        
    def _process(self):
        return self.input[self.sources[0]]

class ShimExtractor(Extractor):
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        Extractor.__init__(self,needs,nframes,step,key)
    
    @property
    def dim(self):
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
        self.assertEqual([1],data)
    
    def test_in_data_multidim(self):
        re = RootExtractor(shape=2)
        si = SingleInput(re)
        re.collect()
        re.process()
        si.collect()
        data = np.array(si.in_data)
        self.assertTrue(data is not None)
        compare = np.array([[1,1]])
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
        v = d[re]
        self.assertEqual(10,len(v))
        self.assertTrue(all([q == 1 for q in v]))
        
        
    def test_two_extractor_chain_no_step(self):
        re = RootExtractor()
        se = SumExtractor(needs=re,nframes=2,step=1)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        self.assertEqual(2,len(d))
        self.assertTrue(d.has_key(re))
        self.assertTrue(d.has_key(se))
        rev = d[re]
        sev = d[se]
        self.assertEqual(10,len(rev))
        self.assertEqual(9,len(sev))
        self.assertTrue(all([q == 1 for q in rev]))
        self.assertTrue(all([q == 2 for q in sev]))
        
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
        self.assertEqual((10,10),rev.shape)
        self.assertEqual((5,),sev.shape)
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
        sev = np.array(d[se2])
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
        rev = d[re]
        se1v = d[se1]
        se2v = d[se2]
        self.assertEqual(10,len(rev))
        self.assertEqual(9,len(se1v))
        self.assertEqual(8,len(se2v))
        self.assertTrue(all([s == 2 for s in se1v]))
        self.assertTrue(all([s == 4 for s in se2v]))
    
    def test_noop_singledim(self):
        re = RootExtractor()
        se = NoOpExtractor(needs = re)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        inp = np.array(d[re])
        self.assertEqual((10,),inp.shape)
        output = np.array(d[se])
        self.assertEqual((10,1),output.shape)
        
    def test_noop_multidim(self):
        re = RootExtractor(shape=10)
        se = NoOpExtractor(needs = re)
        ec = ExtractorChain([se,re])
        d = ec.collect()
        inp = np.array(d[re])
        self.assertEqual((10,10),inp.shape)
        output = np.array(d[se])
        self.assertEqual((10,1,10),output.shape)
            
        
    
