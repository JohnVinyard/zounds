import unittest
from uuid import uuid4
import os

import numpy as np

from model.frame import Frames,Feature
from analyze.extractor import Extractor
from analyze.feature import FFT,Loudness
from model.pattern import FilePattern
from environment import Environment
from frame import PyTablesFrameController

from analyze.analyze_test import AudioStreamTests


class MockExtractor(Extractor):
        
    def __init__(self,needs=None,key=None):
        Extractor.__init__(self,needs=needs,key=key)
    
    def dim(self,env):
        return (100,)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        raise NotImplemented()


class PyTablesFrameControllerTests(unittest.TestCase):
    
    def setUp(self):
        self.hdf5_file = None
        self.hdf5_dir = None
        self.cleanup = None
        self.to_cleanup = []
    
    def tearDown(self):
        if self.cleanup:
            self.cleanup()
        
        for c in self.to_cleanup:
            try:
                os.remove(c)
            except IOError:
                # the file has already been removed
                pass
    
    def make_sndfile(self,length_in_samples,env):
        fn = AudioStreamTests.make_sndfile(length_in_samples,
                                           env.windowsize,
                                           env.samplerate)
        self.to_cleanup.append(fn)
        return fn
    
    def cwd(self):
        return os.getcwd()
    
    def unique(self):
        return str(uuid4())
    
    def cleanup_hdf5_file(self):
        os.remove(os.path.join(self.cwd(),self.hdf5_file))
        
    def cleanup_hdf5_dir(self):
        os.remove(os.path.join(self.cwd(),
                               self.hdf5_dir,
                               self.hdf5_file))
        os.rmdir(os.path.join(self.cwd(),self.hdf5_dir))
        
        
    def hdf5_filename(self):
        self.hdf5_file = '%s.h5' % self.unique()
        self.cleanup = self.cleanup_hdf5_file
        return self.hdf5_file
    
    def hdf5_filepath(self):
        self.hdf5_dir = self.unique()
        self.hdf5_file = '%s.h5' % self.unique()
        self.cleanup = self.cleanup_hdf5_dir
        return '%s/%s' % (self.hdf5_dir,self.hdf5_file)
    
    class AudioConfig:
        samplerate = 44100
        windowsize = 4096
        stepsize = 2048
    
    def FM(self,indir = False,audio_config = AudioConfig,framemodel = None):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        fn = self.hdf5_filepath() if indir else self.hdf5_filename()
        FM = FM1 if not framemodel else framemodel
        Environment('test',
                    FM,
                    PyTablesFrameController,
                    (FM,fn),
                    {},
                    audio_config)
        return fn,FM
    
    def test_file_exists(self):
        fn,FM1 = self.FM()
        self.assertTrue(os.path.exists(fn))
        FM1.controller().close()
    
    def test_file_exists_with_path(self):
        fn,FM1 = self.FM(indir = True)
        self.assertTrue(os.path.exists(fn))
        FM1.controller().close()
    
    def test_read_instance_not_null(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue(c.db_read is not None)
        c.close()
        
    def test_correct_num_columns(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue(len(c.db_read.cols) > 3)
    
    def test_cols_col_shape(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertEqual((0,2048),c.db_read.cols.fft.shape)
        self.assertEqual((0,),c.db_read.cols.loudness.shape)
    
    def test_cols_index(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue(c.db_read.cols.loudness.index is not None)
        self.assertTrue(c.db_read.cols.fft.index is None)
    
    
    def test_cols_dtype(self):
        
        class FrameModel(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            mock = Feature(MockExtractor,store=True,needs=loudness)
        
        fn,FM1 = self.FM(framemodel = FrameModel)
        c = FM1.controller()
        self.assertEqual('float32',c.db_read.cols.mock.dtype)
    
    def test_unstored_col(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
        
        fn,FM = self.FM(framemodel = FM1)
        c = FM.controller()
        print c.db_read.colnames
        self.assertTrue('loudness' not in c.db_read.colnames)
    
    def test_audio_column(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue('audio' in c.db_read.colnames)
    
    def get_patterns(self,framemodel,lengths):
        p = []
        env = framemodel.env()
        for i,l in enumerate(lengths):
            fn = self.make_sndfile(l,env)
            _id = str(i)
            p.append(FilePattern(_id,'test',_id,fn))
        return p
    
    def test_len(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        l = FM1.env().windowsize
        fn = self.make_sndfile(l,FM1.env())
        p = FilePattern('0','test','0',fn)
        ec = FM1.extractor_chain(p)
        c.append(ec)
        self.assertEqual(2,len(c))
    
    def test_list_ids(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        lengths = [2048] * 2
        patterns = self.get_patterns(FM1, lengths)
        for p in patterns:
            ec = FM1.extractor_chain(p)
            c.append(ec)
        _ids = c.list_ids()
        self.assertEqual(2,len(_ids))
        self.assertTrue('0' in _ids)
        self.assertTrue('1' in _ids)
        
    def test_append(self):
        self.fail()
    
    
        
    
