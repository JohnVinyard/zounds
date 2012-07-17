import unittest
from uuid import uuid4
import os
import numpy as np

from zounds.analyze.test_analyze import AudioStreamTests
from zounds.analyze.feature.spectral import FFT,Loudness
from zounds.data.frame import PyTablesFrameController
from zounds.model.frame import Feature,Frames
from zounds.model.pattern import FilePattern
from zounds.environment import Environment

from fetch import PrecomputedFeature

class PrecomputedFeatureTests(unittest.TestCase):
    
    
    def setUp(self):
        pytables_fn = '%s.h5' % uuid4().hex
        self.implementations = [(PyTablesFrameController,(pytables_fn,))]
        self.to_cleanup = [pytables_fn]
        Environment._test = True

    
    def tearDown(self):
        for tc in self.to_cleanup:
            try:
                os.remove(tc)
            except OSError:
                pass
        Environment._test = False
    
    class FrameModel(Frames):
        fft = Feature(FFT, store = True, needs = None)
        loudness = Feature(Loudness, store = True, needs = fft)
    
    class AudioConfig:
        samplerate = 44100
        windowsize = 2048
        stepsize = 1024
        window = None
    
    def append_files(self,
                     framecontroller,
                     framecontroller_args,
                     framemodel = FrameModel,
                     file_lengths = [44100, 44100 * 1.3]):
        
        env = Environment('test',
                          framemodel,
                          framecontroller,
                          tuple([framemodel] + list(framecontroller_args)),
                          {},
                          audio = PrecomputedFeatureTests.AudioConfig)
        
        msf = AudioStreamTests.make_sndfile
        filenames = [msf(fl,env.windowsize,env.samplerate)\
                      for fl in file_lengths]
        self.to_cleanup.extend(filenames)
        for i,fn in enumerate(filenames):
            _id = str(i)
            fp = FilePattern(_id,'test',_id,fn)
            ec = framemodel.extractor_chain(fp)
            framemodel.controller().append(ec)
        
        return env,framemodel,filenames
    
    def test_data_proper_shape_oned(self):
        for i in self.implementations:
            env,framemodel,filenames = self.append_files(i[0], i[1])
            pc = PrecomputedFeature(1,
                                    PrecomputedFeatureTests.FrameModel.loudness)
            data = pc(nexamples = 10)
            self.assertEqual((10,),data.shape)
    
    def test_data_proper_shape_oned_nframes_2(self):
        for i in self.implementations:
            env,framemodel,filenames = self.append_files(i[0], i[1])
            pc = PrecomputedFeature(2,
                                    PrecomputedFeatureTests.FrameModel.loudness)
            data = pc(nexamples = 10)
            self.assertEqual((10,2),data.shape)
            
    def test_data_proper_shape_twod(self):              
        for i in self.implementations:
            env,framemodel,filenames = self.append_files(i[0], i[1])
            pc = PrecomputedFeature(1,
                                    PrecomputedFeatureTests.FrameModel.fft)
            data = pc(nexamples = 11)
            expected_axis1 = PrecomputedFeatureTests.AudioConfig.windowsize / 2
            self.assertEqual((11,expected_axis1),data.shape)
    
    def test_data_proper_shape_twod_nframes_2(self):
        for i in self.implementations:
            env,framemodel,filenames = self.append_files(i[0], i[1])
            pc = PrecomputedFeature(2,
                                    PrecomputedFeatureTests.FrameModel.fft)
            data = pc(nexamples = 11)
            expected_axis1 = PrecomputedFeatureTests.AudioConfig.windowsize / 2
            self.assertEqual((11,expected_axis1*2),data.shape)
            
    def test_pattern_shorter_than_nframes(self):
        for i in self.implementations:
            env,framemodel,filenames = \
                self.append_files(i[0], i[1],file_lengths = [2048,44100])
            pc = PrecomputedFeature(3,
                                    PrecomputedFeatureTests.FrameModel.fft)
            data = pc(nexamples = 20)
            expected_axis1 = PrecomputedFeatureTests.AudioConfig.windowsize / 2
            self.assertEqual((20,expected_axis1*3), data.shape)
    
    def test_pattern_too_many_frames(self):
        for i in self.implementations:
            env,framemodel,filenames = \
                  self.append_files(i[0], i[1],file_lengths = [2048,44100])
            pc = PrecomputedFeature(3,
                                  PrecomputedFeatureTests.FrameModel.fft)
            self.assertRaises(ValueError,lambda : pc(nexamples = 2000000))
            
    def test_with_reduction(self):
        for i in self.implementations:
            env,framemodel,filenames = \
                  self.append_files(i[0], i[1])
            pc = PrecomputedFeature(\
                        3,PrecomputedFeatureTests.FrameModel.fft,reduction = np.max)
            samples = pc(10)
            self.assertEqual((10,1024),samples.shape)
    