import unittest
from uuid import uuid4
import os

from model.frame import Frames,Feature
from analyze.feature import FFT,Loudness
from environment import Environment
from frame import PyTablesFrameController

class PyTablesFrameControllerTests(unittest.TestCase):
    
    def setUp(self):
        self.hdf5_file = None
        self.hdf5_dir = None
        self.cleanup = None
    
    def tearDown(self):
        if self.cleanup:
            self.cleanup()
    
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
    
    def test_file_exists(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        fn = self.hdf5_filename()
        Environment('test',
                    FM1,
                    PyTablesFrameController,
                    (FM1,fn),
                    {})
        self.assertTrue(os.path.exists(fn))
        FM1.controller().close()
    
    def test_file_exists_with_path(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        fn = self.hdf5_filepath()
        Environment('test',
                    FM1,
                    PyTablesFrameController,
                    (FM1,fn),
                    {})
        self.assertTrue(os.path.exists(fn))
        FM1.controller().close()
    
