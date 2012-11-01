import unittest
from zounds.environment import Environment
from zounds.testhelper import make_sndfile,remove,filename
from zounds.model.frame import Frames,Feature
from zounds.model.pattern import FilePattern
from zounds.analyze.feature.spectral import FFT,BarkBands

from zounds.data.frame.filesystem import FileSystemFrameController

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
    
    def set_up(self):
        self.to_cleanup = []
        Environment._test = True
        
        # setup the environment
        dr = filename(extension = '')
        self.env = Environment('Test',
                               FrameModel,
                               FileSystemFrameController,
                               (FrameModel,dr),
                               {Zound2 : self._pattern_controller},
                               audio = AudioConfig)
        self.to_cleanup.append(dr)
        
        # create a single audio file
        fn = make_sndfile(\
                    44100 * 2, AudioConfig.windowsize,AudioConfig.samplerate)
        self._pattern_id = 'ID'
        
        # analyze the file
        fp = FilePattern(self._pattern_id,'Test',self._pattern_id,fn)
        ec = FrameModel.extractor_chain(fp)
        self.env.framecontroller.append(ec)
        self.to_cleanup.append(fn)
    
    
    def tear_down(self):
        for c in self.to_cleanup:
            remove(c)
        Environment._test = False
    
    def test_bad_id(self):
        self.assertRaises(KeyError,lambda : Zound2['BAD_ID'])
    
    def test_bad_id_list(self):
        self.assertRaises(KeyError,lambda : Zound2[['BAD_ID_1,BAD_ID2']])
    
    def test_good_id(self):
        z = Zound2[self._pattern_id]
        self.assertTrue(isinstance(z,Zound2))
    
    def test_good_id_list(self):
        # create a single audio file
        fn = make_sndfile(\
                    44100 * 1, AudioConfig.windowsize,AudioConfig.samplerate)
        _id = 'ID2'
        
        # analyze the file
        fp = FilePattern(_id,'Test',_id,fn)
        ec = FrameModel.extractor_chain(fp)
        self.env.framecontroller.append(ec)
        self.to_cleanup.append(fn)
        
        z = Zound2[[self._pattern_id,_id]]
        self.assertEqual(2,len(z))
        self.assertTrue(all([isinstance(x,Zound2) for x in z]))


from zounds.data.pattern import InMemory
from zounds.model.pattern import Zound2

class InMemoryTest(unittest.TestCase,PatternTest):
    
    def setUp(self):
        self._pattern_controller = InMemory()
        self.set_up()
    
    def tearDown(self):
        self.tear_down()

    