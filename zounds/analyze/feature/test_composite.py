from zounds.analyze.extractor import Extractor,ExtractorChain
from composite import Composite
import numpy as np
import unittest
from zounds.testhelper import RootExtractor

class SourceData(Extractor):
    
    def __init__(self, needs = None, key = None, nframes = 1, step = 1, value = 1):
        Extractor.__init__(self, needs = needs, key = key, 
                           nframes = nframes, step = step)
        self.value = value
        self._dim = np.array(self.value).shape 
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return np.array(self.value).dtype
    
    def __hash__(self):
        return hash(\
            (self.__class__.__name__,
             self.key,
             frozenset(self.sources),
             self.nframes,
             self.step,
             self.value))
    
    def _process(self):
        frames = max([self.input[src].shape[0] for src in self.sources])
        a = np.ndarray((frames,) + self._dim)
        a[...] = self.value
        return a
        

class CompositeTests(unittest.TestCase):
    
    
    def make_chain(self,v1,v2,nframes = 1):
        root = RootExtractor()
        sd0 = SourceData(value = v1, needs = root)
        sd1 = SourceData(value = v2, needs = root)
        c = Composite(needs = [sd0,sd1], step = nframes, nframes = nframes)
        ec = ExtractorChain([root,sd0,sd1,c])
        return c,ec
    
    def extract(self,ec,c):
        return np.concatenate(ec.collect()[c])
        
    def test_both_oned(self):
        c,ec = self.make_chain(1,2)
        self.assertEqual(2,c.dim(None))
        data = self.extract(ec,c)
        self.assertEqual((10,2),data.shape)
        self.assertTrue(np.all([1,2] == data[0]))
    
    def test_different_dimensions(self):
        c,ec = self.make_chain(1,(10,20))
        self.assertEqual(3,c.dim(None))
        data = self.extract(ec, c)
        self.assertEqual((10,3),data.shape)
        self.assertTrue(np.all([1,10,20] == data[0]))
    
    def test_both_twod(self):
        c,ec = self.make_chain((10,20),(30,40))
        self.assertEqual(4,c.dim(None))
        data = self.extract(ec, c)
        self.assertEqual((10,4),data.shape)
        self.assertTrue(np.all([10,20,30,40] == data[0]))
    
    def test_both_oned_two_frames(self):
        c,ec = self.make_chain(1, 2, nframes = 2)
        self.assertEqual(4,c.dim(None))
        data = self.extract(ec, c)
        self.assertEqual((5,4),data.shape)
        self.assertTrue(np.all([1,1,2,2] == data[0]))
    
    def test_different_dimensions_two_frames(self):
        c,ec = self.make_chain(1, (10,20), nframes = 2)
        self.assertEqual(6,c.dim(None))
        data = self.extract(ec, c)
        self.assertEqual((5,6),data.shape)
        self.assertTrue(np.all([1,1,10,20,10,20] == data[0]))
    
    def test_both_twod_twoframes(self):
        c,ec = self.make_chain((10,20), (30,40), nframes = 2)
        self.assertEqual(8,c.dim(None))
        data = self.extract(ec, c)
        self.assertEqual((5,8),data.shape)
        self.assertTrue(np.all([10,20,10,20,30,40,30,40] == data[0]))
        
        
        
        
