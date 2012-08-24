from __future__ import division
import unittest
from uuid import uuid4
import os

import numpy as np

from zounds.model.frame import Frames,Feature,Precomputed
from zounds.analyze.extractor import Extractor,SingleInput
from zounds.analyze.feature.spectral import \
    FFT,Loudness,SpectralCentroid,BarkBands
from zounds.analyze.feature.basic import UnitNorm,Abs
from zounds.analyze.feature.reduce import Downsample
from zounds.model.pattern import FilePattern
from zounds.environment import Environment
from frame import FrameController
from pytables import PyTablesFrameController
from filesystem import FileSystemFrameController
from zounds.testhelper import make_sndfile,remove,SumExtractor


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


class Count(SingleInput):
    def __init__(self,needs = None, key = None, step = 10, nframes = 10):
        SingleInput.__init__(self, needs = needs, key = key, step = step, nframes = nframes)
        self._n = 0
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        n = self._n
        self._n += 1
        return n

class FrameControllerTests(object):
    
    def __init__(self):
        object.__init__(self)
    
    def set_up(self):
        self.to_cleanup = []
        Environment._test = True
    
    def tear_down(self):
        for c in self.to_cleanup:
            remove(c)
        Environment._test = False
    
    def make_sndfile(self,length_in_samples,env):
        fn = make_sndfile(length_in_samples,env.windowsize,env.samplerate)
        self.to_cleanup.append(fn)
        return fn
    
    def cwd(self):
        return os.getcwd()
    
    def unique(self):
        return str(uuid4())
    
    def _db_filepath(self,indir):
        raise NotImplemented()
    
    def db_filepath(self,indir = False):
        dbfp = self._db_filepath(indir = indir)
        self.to_cleanup.append(dbfp)
        return dbfp
    
    def get_patterns(self,framemodel,lengths):
        p = []
        env = framemodel.env()
        for i,l in enumerate(lengths):
            fn = self.make_sndfile(l,env)
            _id = str(i)
            p.append(FilePattern(_id,'test',_id,fn))
        return p

    def build_with_model(self,framemodel,close_db = True):
        fn,FM1 = self.FM(framemodel = framemodel)
        c = FM1.controller()
        lengths = [44100,44100*2]
        patterns = self.get_patterns(FM1,lengths)
        for p in patterns:
            ec = FM1.extractor_chain(p)
            c.append(ec)
        l1 = len(c)
        old_features = c.get_features()
        if close_db and hasattr(c,'close'):
            c.close()
        return fn,l1,old_features
    
    @property
    def controller_class(self):
        raise NotImplemented()
    
    class AudioConfig:
        samplerate = 44100
        windowsize = 4096
        stepsize = 2048
        window = None
    
    def FM(self,
           indir = False,
           audio_config = AudioConfig,
           framemodel = None,
           filepath = None,
           loudness_frames = 1,
           loudness_step = 1,
           store_loudness = True):
        
        class FM1(Frames):
            fft = Feature(FFT,store=store_loudness,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft,
                               step = loudness_step, nframes = loudness_frames)
        
        if filepath:
            fn = filepath
        else:
            fn = self.db_filepath(indir)
        self.to_cleanup.append(fn)
        
        FM = FM1 if not framemodel else framemodel
        Environment('test',
                    FM,
                    self.controller_class,
                    (FM,fn),
                    {},
                    audio_config)
        return fn,FM
    
    def test_two_chunks_feature_with_step_gt_one_ws_4096_ss_2048(self):
        class AudioConfig:
            samplerate = 44100
            windowsize = 4096
            stepsize = 2048
            window = None
        
        class FM(Frames):
            fft = Feature(FFT,needs = None)
            sm = Feature(SumExtractor,needs = fft,nframes = 60,step = 30)
        
        dbfn = self.db_filepath()
        Z = Environment('test',FM,self.controller_class,(FM,dbfn),{},
                        AudioConfig,chunksize_seconds = 45.)
        sfn = self.make_sndfile(45.6504 * Z.samplerate,Z)
        fp = FilePattern('0','test','0',sfn)
        ec = FM.extractor_chain(fp)
        c = FM.controller()
        try:
            c.append(ec)
        except Exception,e:
            self.fail('%s.append() raised %s' % \
                      (self.controller_class.__name__,str(e)))
    
    def test_two_chunks_feature_with_step_gt_one_ws_2048_ss_1024(self):
        class AudioConfig:
            samplerate = 44100
            windowsize = 2048
            stepsize = 1024
            window = None
        
        class FM(Frames):
            fft = Feature(FFT,needs = None)
            sm = Feature(SumExtractor,needs = fft,nframes = 60,step = 30)
        
        dbfn = self.db_filepath()
        Z = Environment('test',FM,self.controller_class,(FM,dbfn),{},
                        AudioConfig,chunksize_seconds = 45.)
        sfn = self.make_sndfile(45.6504 * Z.samplerate,Z)
        fp = FilePattern('0','test','0',sfn)
        ec = FM.extractor_chain(fp)
        c = FM.controller()
        try:
            c.append(ec)
        except Exception,e:
            self.fail('%s.append() raised %s' % \
                      (self.controller_class.__name__,str(e)))
    
    def test_len(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        l = FM1.env().windowsize
        # create a sndfile that is one windowsize long
        fn = self.make_sndfile(l,FM1.env())
        p = FilePattern('0','test','0',fn)
        ec = FM1.extractor_chain(p)
        c.append(ec)
        self.assertEqual(1,len(c))
    
    def test_nframes_and_step_size_disagree_crosses_buffer_boundary(self):
        fn,FM1 = self.FM(loudness_step = 10,loudness_frames = 20)
        c = FM1.controller()
        chunksize = Environment.instance.chunksize 
        l = chunksize * 2
        fn = self.make_sndfile(l,FM1.env())
        p = FilePattern('0','test','0',fn)
        ec = FM1.extractor_chain(p)
        c.append(ec)
        loudness = c['0']['loudness']
        self.assertTrue(np.all(loudness[chunksize - 20 : chunksize + 20] > 0))
    
    
    def test_nframes_and_step_size_disagree_crosses_buffer_boundary_compound_nframes(self):
        class FM(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft,
                               step = 10, nframes = 20)
            l2 = Feature(Loudness, store = True, needs = loudness,
                         step = 1, nframes = 1)
        
        
        fn = self.db_filepath()
        Environment('test',
                    FM,
                    self.controller_class,
                    (FM,fn),
                    {},
                    self.AudioConfig)
        
        c = FM.controller()
        chunksize = Environment.instance.chunksize 
        l = chunksize * 2
        fn = self.make_sndfile(l,FM.env())
        p = FilePattern('0','test','0',fn)
        ec = FM.extractor_chain(p)
        c.append(ec)
        l2 = c['0']['l2']
        # ensure that values spanning the chunk don't get zeroed
        self.assertTrue(np.all(l2[chunksize - 20 : chunksize + 20] > 0))
    
    
    def test_iter_feature_step_greater_pattern_length(self):
        class AudioConfig:
            samplerate = 44100
            windowsize = 2048
            stepsize = 1024
            window = None
        
        class FM(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft,
                               step = 30, nframes = 60)
        
        fn = self.db_filepath()
        Environment('test',
                    FM,
                    self.controller_class,
                    (FM,fn),
                    {},
                    AudioConfig,
                    chunksize_seconds = 45.)
        c = FM.controller()
        env = FM.env()
        fn = self.make_sndfile(.35 * AudioConfig.samplerate, env)
        p = FilePattern('0','test','0',fn)
        ec = FM.extractor_chain(p)
        c.append(ec)
        l = [f for f in c.iter_feature('0',FM.loudness,step = 30,
                                       chunksize = env.chunksize_frames)]
        
        self.assertEqual(1,len(l))
    
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
    
    
    def test_external_id(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        lengths = [5000]
        patterns = self.get_patterns(FM1,lengths)
        ec = FM1.extractor_chain(patterns[0])
        c.append(ec)
        p = patterns[0]
        src,extid = c.external_id(p._id)
        self.assertEqual(src,p.source)
        self.assertEqual(extid,p.external_id)
    
    def test_get_dtype_str(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertEqual('float32',c.get_dtype('loudness'))
        
    def test_get_dtype_feature(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertEqual('float32',c.get_dtype(FM1.loudness))
    
    def test_get_dim_str(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        fftsize = FM1.env().windowsize / 2
        self.assertEqual((fftsize,),c.get_dim('fft'))
        
    def test_get_dim_feature(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        fftsize = FM1.env().windowsize / 2
        self.assertEqual((fftsize,),c.get_dim(FM1.fft))
    
    def test_iter_feature(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        lengths = [FM1.env().windowsize]
        p = self.get_patterns(FM1,lengths)[0]
        ec = FM1.extractor_chain(p)
        c.append(ec)
        
        framens = []
        for fn in c.iter_feature(p._id,'framen'):
            framens.append(fn)
        
        self.assertEqual(1,len(c))
        self.assertEqual(1,len(framens))
        self.assertTrue(0 in framens)
    
    def test_get_features(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        features = c.get_features()
        self.assertEqual(6,len(features))
        self.assertTrue('loudness' in features)
        self.assertTrue('fft' in features)
    
    def test_get_id(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        lengths = [44100,44100*2]
        patterns = self.get_patterns(FM1,lengths)
        for p in patterns:
            ec = FM1.extractor_chain(p)
            c.append(ec)
        
        data = c.get('0')
        self.assertTrue(len(data) > 0)
        self.assertEqual(1,len(set(data['_id'])))
    
    def test_get_source_and_ext_id(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        lengths = [44100,44100*2]
        patterns = self.get_patterns(FM1,lengths)
        for p in patterns:
            ec = FM1.extractor_chain(p)
            c.append(ec)
        
        data = c.get(('test','0'))
        self.assertTrue(len(data) > 0)
        self.assertEqual(1,len(set(data['_id'])))
    
    def test_get_address(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        lengths = [44100,44100*2]
        patterns = self.get_patterns(FM1,lengths)
        addresses = []
        for p in patterns:
            ec = FM1.extractor_chain(p)
            addresses.append(c.append(ec))
            
        data = c.get(addresses[0])
        self.assertTrue(len(data) > 0)
        self.assertEqual(1,len(set(data['_id'])))
    
    def test_get_invalid_key(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertRaises(ValueError,lambda : c.get(dict()))
    
    def sync_helper(self,old_framemodel,new_framemodel,*assertions):
        
        fn,l1,old_features = self.build_with_model(old_framemodel)
        
        FM2 = new_framemodel
        # make sure to use the same file
        fn,FM2 = self.FM(framemodel = FM2,filepath = fn)
        FM2.sync()
        c = FM2.controller()
        self.assertEqual(l1,len(c))
        features = c.get_features()
        
        for a in assertions:
            self.assertTrue(a(old_features,features))
    
    def test_sync_add_feature(self):
        
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            l2 = Feature(Loudness,store=True,needs=fft,nframes=2)
            
        a1 = lambda old,new : 'l2' in new
        a2 = lambda old,new : len(old) + 1 == len(new)
        self.sync_helper(FM1,FM2,a1,a2)
    
    def test_sync_delete_feature(self):
        
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
        
        a1 = lambda old,new : 'loudness' not in new
        a2 = lambda old,new : len(old) - 1 == len(new)
        self.sync_helper(FM1,FM2,a1,a2)
    
    def test_sync_unchanged(self):
        
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        a1 = lambda old,new : old == new
        self.sync_helper(FM1,FM2,a1)
        
    def test_sync_store(self):
        
        # loudness is NOT stored
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
        
        # loudness is stored
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        self.sync_helper(FM1, 
                         FM2,
                         lambda old,new : len(old) == len(new),
                         lambda old,new : not old['loudness'].store,
                         lambda old,new : new['loudness'].store)
        
    def test_sync_unstore(self):
        
        # loudness is stored
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        # loudness is NOT stored
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
        
        
        self.sync_helper(FM1,
                         FM2,
                         lambda old,new : len(old) == len(new),
                         lambda old,new : old['loudness'].store,
                         lambda old,new : not new['loudness'].store)
        
                         
    
    def test_sync_changed_feature(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
        
        
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft,nframes=2)
        
        self.sync_helper(FM1,
                         FM2,
                         lambda old,new : len(old) == len(new),
                         lambda old,new : old != new,
                         lambda old,new : 'loudness' in old,
                         lambda old,new : 'loudness' in new)
        
    def test_sync_2d_feature_with_stepsize(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            downsample = Feature(Downsample,needs = fft, store = True,
                                 inshape = (6,2048), factor = 2, 
                                 step = 6, nframes = 6)
        
        
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            downsample = Feature(Downsample,needs = fft, store = True,
                                 inshape = (6,2048), factor = 2, 
                                 step = 6, nframes = 6)
            centroid = Feature(SpectralCentroid,store = True, needs = fft)
        
        self.sync_helper(FM1,FM2,
                         lambda old,new : len(old) != len(new),
                         lambda old,new : 'downsample' in new,
                         lambda old,new : 'centroid' in new)
    
    
            
    
    def test_feature_with_stepsize(self):
        
        
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            count = Feature(Count,store = True, needs = None)
        
        
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            count = Feature(Count,store = True, needs = None)
            centroid = Feature(SpectralCentroid,store = True, needs = fft)
        
        fn,l1,old_features = self.build_with_model(FM1,close_db = False)
        _id = list(FM1.list_ids())[0]
        orig_count = FM1[_id].count
        FM1.controller().close()
        
        # make sure to use the same file
        fn,FM2 = self.FM(framemodel = FM2,filepath = fn)
        FM2.sync()
        c = FM2.controller()
        self.assertEqual(l1,len(c))
        self.assertTrue(np.all(FM2[_id].count == orig_count))
        
        
        
    
    def test_sync_non_stored_ancestor(self):
        class FM1(Frames):
            fft = Feature(FFT,store=False,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            centroid = Feature(SpectralCentroid,store=True,needs=fft)
        
        fn,l1,old_features = self.build_with_model(FM1)
        
        fn,FM2 = self.FM(framemodel = FM1, filepath = fn)
        add,update,delete,recompute = FM2._sync()
        self.assertFalse(add)
        self.assertFalse(recompute)
    
    def test_sync_new_feature_depends_on_non_stored_ancestor(self):
        
        class FM1(Frames):
            fft = Feature(FFT,store=False,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=False,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            centroid = Feature(SpectralCentroid,store=True,needs=fft)
        
        fn,l1,old_features = self.build_with_model(FM1)
        
        fn,FM2 = self.FM(framemodel = FM2, filepath = fn)
        add,update,delete,recompute = FM2._sync()
        self.assertTrue('centroid' in add)
        self.assertEqual(2,len(recompute))
        
        chain = FM2.extractor_chain(FilePattern('0','0','0','/some/file.wav'),
                                    transitional = True,
                                    recompute = recompute)
        self.assertTrue(isinstance(chain['loudness'],Precomputed))
        self.assertTrue(isinstance(chain['centroid'],SpectralCentroid))
        self.assertTrue(isinstance(chain['fft'],FFT))
    
    def test_unstored_with_stored_ancestor(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        class FM2(Frames):
            fft = Feature(FFT,store=False,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            
        fn,l1,old_features = self.build_with_model(FM1)
        
        fn,FM2 = self.FM(framemodel = FM2, filepath = fn)
        add,update,delete,recompute = FM2._sync()
        
        self.assertTrue('fft' in delete)
        self.assertTrue('fft' not in recompute)
    
    def test_add_unstored_leaf(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            
            
        class FM2(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
            
        fn,l1,old_features = self.build_with_model(FM1)
        
        fn,FM2 = self.FM(framemodel = FM2, filepath = fn)
        add,update,delete,recompute = FM2._sync()
        self.assertFalse(add)
        self.assertFalse(update)
        self.assertFalse(delete)
        self.assertFalse(recompute)
        
        
    def test_sync_unstored_unchanged_feature_in_lineage(self):
        
        class FM1(Frames):
            fft = Feature(FFT, needs = None, store = False)
            bark = Feature(BarkBands, needs = fft, store = True, nbands = 100)
            barkun = Feature(UnitNorm, needs = bark, inshape = 100, store = False)
            loud = Feature(Loudness, needs = barkun, store = True)
            abs = Feature(Abs, needs = loud, store = True, inshape = ())
        
        class FM2(Frames):
            fft = Feature(FFT, needs = None, store = False)
            bark = Feature(BarkBands, needs = fft, store = True, nbands = 100)
            barkun = Feature(UnitNorm, needs = bark, inshape = 100, store = False)
            loud = Feature(Loudness, needs = barkun, store = True)

        self.sync_helper(FM1,FM2,lambda old,new : 'abs' not in new)
                        
                        
    def test_sync_ancestor_feature_recomputed(self):
        '''
        This failing test results in an extractor chain with a Precomputed
        FFT feature in the extractor chain. In this case, we'd like to simply
        delete loudness, copy centroid, and do nothing with FFT.
        '''
        class FM1(Frames):
            fft = Feature(FFT,store = False, needs = None)
            loudness = Feature(Loudness,store = True, needs = fft)
            centroid = Feature(SpectralCentroid,store = True, needs = fft)
        
        class FM2(Frames):
            fft = Feature(FFT,store = False, needs = None)
            centroid = Feature(SpectralCentroid,store = True, needs = fft)
            
        fn,l1,old_features = self.build_with_model(FM1)
        fn,FM2 = self.FM(framemodel = FM2, filepath = fn)
        add,update,delete,recompute = FM2._sync()
        self.assertTrue('loudness' in delete)
        self.assertTrue('centroid' not in recompute)
        
        chain = FM2.extractor_chain(FilePattern('0','0','0','/some/file.wav'),
                                    transitional = True,
                                    recompute = recompute)
        self.assertTrue(isinstance(chain['centroid'],Precomputed))
        self.sync_helper(FM1,
                         FM2,
                         lambda old,new : len(old) != len(new),
                         lambda old,new : 'loudness' in old,
                         lambda old,new : 'loudness' not in new)

        
    def test_iter_all_step_size_1(self):
        
        class AudioConfig:
            samplerate = 44100
            windowsize = 2048
            stepsize = 1024
            window = None
        
        class FM(Frames):
            fft = Feature(FFT,store = False, needs = None)
            loudness = Feature(Loudness,store = True, needs = fft)
            centroid = Feature(SpectralCentroid,store = True, needs = fft)
        
        fn,FM1 = self.FM(framemodel = FM, audio_config = AudioConfig)
        c = FM.controller()
        lengths = [44100,44100*2]
        patterns = self.get_patterns(FM,lengths)
        for p in patterns:
            ec = FM1.extractor_chain(p)
            c.append(ec)
        total_length = len(c)
        
        f = []
        _ids = []
        for address,frames in c.iter_all(step = 1):
            _ids.append(frames._id)
            f.append((address,frames))
        
        self.assertTrue(all([isinstance(frm,FM) for a,frm in f]))
        self.assertEqual(total_length,len(f))
        self.assertEqual(2,len(set([_id for _id in _ids])))
    
    def test_iter_all_step_size_2(self):
        
        class AudioConfig:
            samplerate = 44100
            windowsize = 2048
            stepsize = 1024
            window = None
        
        class FM(Frames):
            fft = Feature(FFT,store = False, needs = None)
            loudness = Feature(Loudness,store = True, needs = fft)
            centroid = Feature(SpectralCentroid,store = True, needs = fft)
        
        fn,FM1 = self.FM(framemodel = FM, audio_config = AudioConfig)
        c = FM.controller()
        lengths = [44100,44100*2]
        patterns = self.get_patterns(FM,lengths)
        for p in patterns:
            ec = FM1.extractor_chain(p)
            c.append(ec)
        total_length = len(c)
        
        f = []
        _ids = []
        for address,frames in c.iter_all(step = 2):
            f.append((address,frames))
            # we must call extend, since the frames instances will
            # have a length greater than 1
            _ids.extend(frames._id)
        
        self.assertTrue(all([isinstance(frm,FM) for a,frm in f]))
        self.assertEqual(2,len(set([_id for _id in _ids])))
        self.assertNotEqual(total_length,len(f))
        # Not all of the slices will have length two, because of odd frame
        # counts.  Just make sure one of them does.
        self.assertEqual(2,len(f[0][1]))

class PyTablesFrameControllerTests(unittest.TestCase,FrameControllerTests):
    
    def setUp(self):
        self.set_up()
    
    def tearDown(self):
        self.tear_down()
    
    def _db_filepath(self,indir):
        if indir:
            self.hdf5_dir = self.unique()
            self.hdf5_file = '%s.h5' % self.unique()
            self._path = '%s/%s' % (self.hdf5_dir,self.hdf5_file)
        else:
            self.hdf5_file = '%s.h5' % self.unique()
            self._path = self.hdf5_file
        return self._path
    
    @property
    def controller_class(self):
        return PyTablesFrameController
    
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
        self.assertTrue('loudness' not in c.db_read.colnames)
    
    def test_audio_column(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue('audio' in c.db_read.colnames)
    
class FileSystemFrameControllerTests(unittest.TestCase,FrameControllerTests):
    
    def setUp(self):
        self.set_up()
    
    def tearDown(self):
        self.tear_down()
    
    def _db_filepath(self,indir):
        if indir:
            self._path = '%s/%s' % (self.unique(),self.unique())
        else:
            self._path = self.unique()
        
        return self._path
    @property
    def controller_class(self):
        return FileSystemFrameController
    
    def test_file_exists(self):
        fn,FM1 = self.FM()
        self.assertTrue(os.path.exists(fn))
    
    def test_file_exists_with_path(self):
        fn,FM1 = self.FM(indir = True)
        self.assertTrue(os.path.exists(fn))
        FM1.controller().close()
        
    def test_correct_num_columns(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue(len(c._np_dtype) > 3)
    
    def test_cols_col_shape(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertEqual((0,2048),c.get_dim('fft'))
        self.assertEqual((0,),c.get_dim('loudness'))
    
    def test_cols_dtype(self):
        
        class FrameModel(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=True,needs=fft)
            mock = Feature(MockExtractor,store=True,needs=loudness)
        
        fn,FM1 = self.FM(framemodel = FrameModel)
        c = FM1.controller()
        self.assertEqual(np.float32,c.get_dtype('mock'))
    
    def test_unstored_col(self):
        class FM1(Frames):
            fft = Feature(FFT,store=True,needs=None)
            loudness = Feature(Loudness,store=False,needs=fft)
        
        fn,FM = self.FM(framemodel = FM1)
        c = FM.controller()
        self.assertRaises(KeyError,lambda : c._np_dtype['loudness'])
    
    def test_audio_column(self):
        fn,FM1 = self.FM()
        c = FM1.controller()
        self.assertTrue(isinstance(c._np_dtype['audio'],np.dtype))
    