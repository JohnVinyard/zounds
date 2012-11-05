import unittest
from zounds.environment import Environment
from zounds.testhelper import make_sndfile,remove,filename
from zounds.model.frame import Frames,Feature
from zounds.model.pattern import FilePattern
from zounds.analyze.feature.spectral import FFT,BarkBands

from zounds.data.frame.filesystem import FileSystemFrameController

from zounds.data.pattern import InMemory
from zounds.model.pattern import Zound,Event

class FrameModel(Frames):
    fft = Feature(FFT)
    bark = Feature(BarkBands,needs = fft)

class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1024
    window = None

class PatternTest(object):
    
    def __init__(self):
        object.__init__(self)
    
    
    def make_leaf_pattern(self,length_seconds,frames_id, store = True):
        
        # create an audio file
        fn = make_sndfile(AudioConfig.samplerate * length_seconds,
                          AudioConfig.windowsize,
                          AudioConfig.samplerate)
        self.to_cleanup.append(fn)
        
        
        
        # analyze it and insert it into the frames database
        src = 'Test'
        fp = FilePattern(frames_id,src,frames_id,fn)
        ec = FrameModel.extractor_chain(fp)
        self.env.framecontroller.append(ec)
        
        # get the address of the pattern
        addr = self.env.framecontroller.address(frames_id)
        
        # create a new leaf pattern
        z = Zound(source = src, external_id = frames_id, _id=frames_id,
                  address = addr, is_leaf = True)
        if store:
            # store it
            z.store()
            return z._id
        
        return z
    
    def set_up(self):
        self.to_cleanup = []
        Environment._test = True
        
        # setup the environment
        dr = filename(extension = '')
        self.env = Environment('Test',
                               FrameModel,
                               FileSystemFrameController,
                               (FrameModel,dr),
                               {Zound : self._pattern_controller},
                               audio = AudioConfig)
        
        
        self.to_cleanup.append(dr)
        self._frame_id = 'ID'
        self._pattern_id = self.make_leaf_pattern(2, self._frame_id)
        return self._pattern_id
        

    def tear_down(self):
        for c in self.to_cleanup:
            remove(c)
        Environment._test = False
    
    def test_bad_id(self):
        self.assertRaises(KeyError,lambda : Zound['BAD_ID'])
    
    def test_bad_id_list(self):
        self.assertRaises(KeyError,lambda : Zound[['BAD_ID_1,BAD_ID2']])
    
    def test_good_id(self):
        z = Zound[self._pattern_id]
        self.assertTrue(isinstance(z,Zound))
    
    def test_recovered(self):
        z = self.make_leaf_pattern(3, 'fid', store = False)
        z.store()
        z2 = Zound[z._id]
        self.assertFalse(z is z2)
        self.assertEqual(z,z2)
    
    def test_good_id_list(self):
        frame_id = 'ID2'
        pattern_id2 = self.make_leaf_pattern(1, frame_id)
        z = Zound[[self._pattern_id,pattern_id2]]
        self.assertEqual(2,len(z))
        self.assertTrue(all([isinstance(x,Zound) for x in z]))
    
    def test_leaf_bad_value(self):
        self.assertRaises(ValueError,lambda : Zound.leaf('BAD ID'))
    
    def test_leaf_frame_id(self):
        z = Zound.leaf(self._frame_id)
        addr = self.env.framecontroller.address(self._frame_id)
        self.assertEqual(addr,z.address)
        self.assertTrue(z.is_leaf)
    
    def test_leaf_addr(self):
        # KLUDGE: This is a FileSystemFrameController specific test
        
        # the first ten frames of the analyzed audio 
        addr = self.env.address_class((self._frame_id,slice(0,10)))
        z = Zound.leaf(addr)
        self.assertEqual(addr,z.address)
        self.assertTrue(z.is_leaf)
    
    def test_frame_instance(self):
        addr = self.env.address_class((self._frame_id,slice(0,10)))
        frames = FrameModel(address = addr)
        z = Zound.leaf(frames)
        self.assertEqual(addr,z.address)
        self.assertTrue(z.is_leaf)
    
    def test_frame_instance_no_address(self):
        # This test demonstrates that sliced Frames-derived instances no longer
        # have an address attribute.  This is really a bug, but it's expected
        # behavior, for now.
        frames = FrameModel.random()[:10]
        self.assertRaises(ValueError,lambda : Zound.leaf(frames))
    
    def test_append(self):
        pid = self.make_leaf_pattern(2, self._frame_id)
        leaf = Zound[pid]
        n = Zound(source = 'Test')
        self.assertFalse(n.is_leaf)
        n.append(leaf,[Event(i) for i in range(4)])
        self.assertTrue(pid in n.all_ids)
        self.assertEqual(4,len(n.data[pid]))
    
    def test_append_nested(self):
        self.fail()
    
    def test_append_stored(self):
        pid = self.make_leaf_pattern(2, self._frame_id)
        leaf = Zound[pid]
        n = Zound(source = 'Test')
        n.append(leaf,[Event(i) for i in range(4)])
        n.store()
        
        r = Zound[n._id]
        self.assertEqual(n,r)
        self.assertFalse(r.is_leaf)
        self.assertTrue(pid in r.all_ids)
        self.assertEqual(4,len(n.data[pid]))
        
        self.assertEqual(r.patterns[pid],leaf)
    
    def test_store_unstored_nested_patterns(self):
        '''
        1) create leaf pattern
        2) append it to a new pattern
        3) when the top-level pattern is stored, ensure that the nested pattern
           is stored too.
        '''
        leaf = self.make_leaf_pattern(2, 'fid', store = False)
        branch = Zound(source = 'Test')
        branch.append(leaf,[Event(i) for i in range(4)])
        branch.store()
        
        r = Zound[branch._id]
        self.assertEqual(branch,r)
        self.assertFalse(r.is_leaf)
        self.assertTrue(leaf._id in r.all_ids)
        self.assertEqual(4,len(r.data[leaf._id]))
        
        lr = Zound[leaf._id]
        self.assertEqual(leaf,lr)
        self.assertTrue(lr.is_leaf)
        
    
    def test_append_stored_nested(self):
        '''
        Similar to test_append_stored(), but with two levels of nesting
        '''
        self.fail()
    
    def test_length_samples(self):
        self.fail()
    
    def test_length_seconds(self):
        self.fail()
    
    def test_length_samples_nested(self):
        self.fail()
    
    def test_length_seconds_nested(self):
        self.fail()
        
        


class InMemoryTest(unittest.TestCase,PatternTest):
    
    def setUp(self):
        self._pattern_controller = InMemory()
        try:
            self.set_up()
        except:
            # KLUDGE: This is a stop-gap solution for when set_up fails.  Get
            # rid of this.
            print 'SETUP FAILED'
    
    def tearDown(self):
        self.tear_down()

    