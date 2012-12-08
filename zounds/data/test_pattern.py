import unittest2
from random import randint
import numpy as np

from zounds.environment import Environment
from zounds.testhelper import make_sndfile,remove,filename
from zounds.model.frame import Frames,Feature
from zounds.model.pattern import FilePattern
from zounds.analyze.feature.spectral import FFT,BarkBands

from zounds.data.frame.filesystem import FileSystemFrameController

from zounds.data.pattern import InMemory,MongoDbPatternController
from zounds.model.pattern import \
    Zound,Event,BaseTransform,ExplicitTransform, \
    CriterionTransform,IndiscriminateTransform, \
    RecursiveTransform,MusicPattern

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
                               {Zound : self._pattern_controller,
                                MusicPattern : self._pattern_controller},
                               audio = AudioConfig)
        
        
        self.to_cleanup.append(dr)
        self._frame_id = 'ID'
        self._pattern_id = self.make_leaf_pattern(2, self._frame_id)
        return self._pattern_id
        

    def tear_down(self):
        for c in self.to_cleanup:
            remove(c)
        Environment._test = False
    
    def test_no_stored_date(self):
        leaf = Zound[self._pattern_id]
        p = Zound(source = 'Test')
        p.append(leaf,[Event(i) for i in range(4)])
        self.assertFalse(p.stored)
    
    def test_stored_date(self):
        leaf = Zound[self._pattern_id]
        p = Zound(source = 'Test')
        p.append(leaf,[Event(i) for i in range(4)])
        p.store()
        print p.stored
        self.assertTrue(p.stored)
        self.assertTrue(isinstance(p.stored,float))
    
    def test_stored_date_retrieved(self):
        leaf = Zound[self._pattern_id]
        p = Zound(source = 'Test')
        p.append(leaf,[Event(i) for i in range(4)])
        p.store()
        _id = p._id
        del p
        
        p = Zound[_id]
        self.assertTrue(p.stored)
        self.assertTrue(isinstance(p.stored,float))
    
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
    
    def test_fetch_by_single_id_and_list(self):
        '''
        demonstrate that fetching a pattern with Zound[_id] and Zound[[_id]]
        yields identical results
        '''
        l1 = Zound[self._pattern_id]
        l2 = Zound[[self._pattern_id]][0]
        self.assertFalse(l1 is l2)
        self.assertTrue(l1 == l2)
    
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
    
    def test_leaf_attributes(self):
        leaf = Zound[self._pattern_id]
        self.assertFalse(leaf.all_ids)
        self.assertFalse(leaf.pdata)
        self.assertFalse(leaf.patterns)
        self.assertFalse(list(leaf.iter_patterns()))
    
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
        self.assertEqual(4,len(n.pdata[pid]))
    
    def test_empty_leaf(self):
        leaf = Zound[self._pattern_id]
        self.assertFalse(leaf.empty)
    
    def test_empty_branch(self):
        b = Zound(source = 'Test')
        self.assertTrue(b.empty)
    
    def test_empty_branch_with_events(self):
        leaf = Zound[self._pattern_id]
        b = Zound(source = 'Test')
        b.append(leaf,[Event(i) for i in range(4)])
        self.assertFalse(b.empty)
    
    def test_store_empty(self):
        '''
        try to store an empty pattern
        '''
        b = Zound(source = 'Test')
        self.assertRaises(Exception,lambda : b.store())
    
    def test_append_after_store(self):
        '''
        try to append() to a pattern that has already been stored
        '''
        pid = self.make_leaf_pattern(2, self._frame_id)
        leaf = Zound[pid]
        n = Zound(source = 'Test')
        n.append(leaf,[Event(i) for i in range(4)])
        n.store()
        self.assertRaises(Exception,lambda : n.append(leaf,[Event(5)]))
    
    def test_remove_after_store(self):
        '''
        try to remove() from a pattern that has already been stored
        '''
        pid = self.make_leaf_pattern(2, self._frame_id)
        leaf = Zound[pid]
        n = Zound(source = 'Test')
        n.append(leaf,[Event(i) for i in range(4)])
        n.store()
        self.assertRaises(Exception,lambda : n.remove())
    
    def test_append_nothing(self):
        '''
        Call append() with an empty list for the events argument
        '''
        pid = self.make_leaf_pattern(2, self._frame_id)
        leaf = Zound[pid]
        n = Zound(source = 'Test')
        self.assertRaises(ValueError,lambda : n.append(leaf,[]))
    
    def test_append_self(self):
        '''
        Attempt to append a pattern to itself
        '''
        p = Zound(source = 'Test')
        self.assertRaises(ValueError,lambda : p.append(p,[Event(i) for i in range(4)])) 
    
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
        self.assertEqual(4,len(n.pdata[pid]))
        
        self.assertEqual(r.patterns[pid],leaf)
    
    def test_append_stored_nested(self):
        '''
        Similar to test_append_stored(), but with two levels of nesting
        '''
        pid = self.make_leaf_pattern(2, self._frame_id)
        leaf = Zound[pid]
        n = Zound(source = 'Test')
        n.append(leaf,[Event(i) for i in range(3)])
        n.store()
        
        root = Zound(source = 'Test')
        root.append(n,[Event(i) for i in range(0,16,4)])
        root.store()
        
        r = Zound[root._id]
        self.assertEqual(root,r)
        self.assertFalse(root is r)
        self.assertTrue(n._id in r.all_ids)
        self.assertTrue(leaf._id in r.all_ids)
        self.assertEqual(4,len(r.pdata[n._id]))
        self.assertEqual(n,r.patterns[n._id])
        
        self.assertEqual(3,len(r.patterns[n._id].pdata[leaf._id]))
        self.assertEqual(leaf,r.patterns[n._id].patterns[leaf._id])
    
    def test_extend_single_pattern_multiple_events(self):
        leaf = Zound[self._pattern_id]
        b = Zound(source = 'Test')
        b.extend(leaf,[Event(i) for i in range(4)])
        
        self.assertEqual(1,len(b.pdata))
        self.assertTrue(leaf._id in b.all_ids)
        self.assertEqual(4,len(b.pdata[leaf._id]))
    
    def test_extend_single_pattern_no_events(self):
        leaf = Zound[self._pattern_id]
        b = Zound(source = 'Test')
        b.extend(leaf,None)
        
        self.assertEqual(0,len(b.pdata))
        self.assertFalse(leaf._id in b.all_ids)
    
    def test_extend_multiple_patterns_one_event_each(self):
        leaf = Zound[self._pattern_id]
        l2 = self.make_leaf_pattern(3, 'fid2', store = False)
        
        b = Zound(source = 'Test')
        b.extend([leaf,l2],[Event(1),Event(2)])
        
        self.assertEqual(2,len(b.pdata))
        self.assertTrue(all([1 == len(e) for e in b.pdata.itervalues()]))
    
    def test_extend_multiple_patterns_one_event(self):
        leaf = Zound[self._pattern_id]
        l2 = self.make_leaf_pattern(3, 'fid2', store = False)
        
        b = Zound(source = 'Test')
        b.extend([leaf,l2],Event(1))
        
        self.assertEqual(2,len(b.pdata))
        self.assertTrue(all([1 == len(e) for e in b.pdata.itervalues()]))
    
    def test_extend_multiple_patterns_multiple_events_each(self):
        leaf = Zound[self._pattern_id]
        l2 = self.make_leaf_pattern(3, 'fid2', store = False)
        
        b = Zound(source = 'Test')
        e1 = [Event(i) for i in range(4)]
        e2 = [Event(i) for i in range(10,12)]
        b.extend([leaf,l2],[e1,e2])
        
        
        self.assertEqual(2,len(b.pdata))
        self.assertTrue(leaf._id in b.all_ids)
        self.assertTrue(l2._id in b.all_ids)
        self.assertEqual(4,len(b.pdata[leaf._id]))
        self.assertEqual(2,len(b.pdata[l2._id]))
    
    def test_multiples_patterns_multiple_events_mismatched_lengths(self):
        leaf = Zound[self._pattern_id]
        l2 = self.make_leaf_pattern(3, 'fid2', store = False)
        
        b = Zound(source = 'Test')
        e1 = [Event(i) for i in range(4)]
        e2 = [Event(i) for i in range(10,12)]
        self.assertRaises(ValueError,lambda : b.extend([leaf,l2],[e1,e2,e2]))
        
    
    def test_store_unstored_nested_patterns(self):
        '''
        A pattern containing unstored patterns should store its "children" as
        well as itself
        '''
        # create a leaf pattern, but don't store it
        leaf = self.make_leaf_pattern(2, 'fid', store = False)
        
        # create a branch pattern
        branch = Zound(source = 'Test')
        # append the leaf pattern a few times
        branch.append(leaf,[Event(i) for i in range(4)])
        # store the branch
        branch.store()
        
        # retrieve the branch and ensure that it looks as expected
        r = Zound[branch._id]
        self.assertEqual(branch,r)
        self.assertFalse(r.is_leaf)
        self.assertTrue(leaf._id in r.all_ids)
        self.assertEqual(4,len(r.pdata[leaf._id]))
        
        # retrieve the leaf. Oops!  The leaf was never stored
        lr = Zound[leaf._id]
        self.assertEqual(leaf,lr)
        self.assertTrue(lr.is_leaf)
    
    def test_store_unstored_nested_patterns_2(self):
        '''
        Just like test_store_unstored_nested_patterns, but nested two levels deep
        '''
        # create a leaf pattern, but don't store it
        leaf = self.make_leaf_pattern(2, 'fid', store = False)
        
        # create a branch pattern
        branch = Zound(source = 'Test')
        
        # append the leaf pattern a few times
        branch.append(leaf,[Event(i) for i in range(4)])
        
        # create the top level pattern
        root = Zound(source = 'Test')
        
        # append the branch pattern a few times
        root.append(branch,[Event(i) for i in range(0,16,4)])
        
        # calling store() on root should store the leaf and branch patterns too
        root.store()
        
        # setUp creates and stores a pattern, so we expect there to be four
        # patterns in the db
        self.assertEqual(1 + 3,len(Zound.controller()))
        
        r2 = Zound[root._id]
        self.assertFalse(r2 is root)
        self.assertEqual(root,r2)
        self.assertEqual(2,len(r2.all_ids))
        self.assertTrue(leaf._id in r2.all_ids)
        self.assertTrue(branch._id in r2.all_ids)
        
        b2 = Zound[branch._id]
        self.assertFalse(b2 is branch)
        self.assertEqual(b2,branch)
        self.assertEqual(1,len(b2.all_ids))
        self.assertTrue(leaf._id in r2.all_ids)
        
        l2 = Zound[leaf._id]
        self.assertFalse(l2 is leaf)
        self.assertEqual(l2,leaf)
        
    # COPY #######################################################
    def _almost_equal(self,z1,z2):
        '''
        Compare copied Zound patterns, ensuring that they're equivalent
        '''
        source = z1.source == z2.source
        addr = z1.address == z2.address
        all_ids = z1.all_ids == z2.all_ids
        leaf = z1.is_leaf == z2.is_leaf
        _to_store = z1._to_store == z2._to_store
        
        if not (source and addr and all_ids and leaf and _to_store):
            return False
        
        for k,v in z1.pdata.iteritems():
            # KLUDGE: This doesn't guarantee equivalence, but it's probably
            # good enough
            if len(z2.pdata[k]) != len(v):
                return False
        
        return True
    
    def test_copy_leaf_stored(self):
        leaf = Zound[self._pattern_id]
        lc = leaf.copy()
        self.assertFalse(lc.stored)
        self.assertFalse(leaf._id == lc._id)
        self.assertTrue(self._almost_equal(leaf, lc))
    
    def test_copy_branch_unstored(self):
        leaf = Zound[self._pattern_id]
        branch = Zound(source = 'Test')
        branch.append(leaf,[Event(i) for i in range(4)])
        b2 = branch.copy()
        self.assertFalse(branch._id == b2._id)
        self.assertTrue(self._almost_equal(b2,branch))
    
    def test_copy_leaf_unstored(self):
        leaf = self.make_leaf_pattern(2, 'fid', store = False)
        lc = leaf.copy()
        self.assertFalse(leaf._id == lc._id)
        self.assertTrue(self._almost_equal(leaf, lc))
    
    def test_copy_branch_stored(self):
        leaf = Zound[self._pattern_id]
        branch = Zound(source = 'Test')
        branch.append(leaf,[Event(i) for i in range(4)])
        branch.store()
        b2 = branch.copy()
        self.assertFalse(b2.stored)
        self.assertFalse(branch._id == b2._id)
        self.assertTrue(self._almost_equal(b2,branch))
    
    def test_music_pattern_copy_rectifies_negative_beats(self):
        leaf = Zound[self._pattern_id]
        mp = MusicPattern(source = 'Test', length_beats = 4, bpm = 60)
        mp.append(leaf,[Event(i) for i in range(4)])
        mp <<= 1
        n = mp.copy()
        self.assertTrue(all([e.time >= 0 for e in n.pdata[leaf._id]]))
    
    ## TRANSFORM ######################################################

    def test_transform_leaf(self):
        '''
        Change the address of a leaf pattern
        '''
        leaf = Zound[self._pattern_id]
        
        def shorten(leaf,events):
            addr = leaf.address
            sl = slice(addr._index.start,addr._index.stop - 5)
            new_addr = self.env.address_class((addr._id,sl))
            leaf.address = new_addr
            return leaf
            
        t = IndiscriminateTransform(shorten)
        nl = leaf.transform(t)
        self.assertFalse(nl is leaf)
        self.assertFalse(nl == leaf)
        self.assertFalse(nl.stored)
        self.assertFalse(leaf.address == nl.address)
    
    def test_transform_branch_remove(self):
        '''
        Remove a leaf pattern entirely from a branch pattern
        '''
        leaf = Zound[self._pattern_id]
        l2_id = self.make_leaf_pattern(2, 'fid2')
        l2 = Zound[l2_id]
        
        branch = Zound(source = 'Test')
        branch.append(leaf,[Event(i) for i in range(4)])
        branch.append(l2,[Event(i) for i in range(6,8)])
        
        def remove(pattern,events):
            return pattern,None
        
        t = ExplicitTransform({leaf._id : remove})
        b2 = branch.transform(t)
        self.assertFalse(b2 is branch)
        self.assertFalse(leaf._id in b2.all_ids)
        self.assertTrue(l2._id in b2.all_ids)
    
    def test_transform_branch_events_to_patterns(self):
        '''
        Alter each occurrence of a pattern so that each of its events becomes a
        new pattern with one event
        '''
        leaf = Zound[self._pattern_id]
        branch = Zound(source = 'Test')
        branch.append(leaf,[Event(i) for i in range(4)])
        
        def change_slice(leaf):
            leaf = leaf.copy()
            addr = leaf.address
            sl = slice(addr._index.start,addr._index.stop - randint(0,10))
            new_addr = self.env.address_class((addr._id,sl))
            leaf.address = new_addr
            return leaf
        
        def alter(pattern,events):
            p = []
            ev = []
            for e in events:
                p.append(change_slice(pattern))
                ev.append(e)
            return p,ev
        
        t = ExplicitTransform({leaf._id : alter})
        b2 = branch.transform(t)
        self.assertFalse(b2 is branch)
        self.assertFalse(b2 == branch)
        self.assertFalse(leaf._id in b2.all_ids)
        self.assertEqual(4,len(b2.pdata))
        self.assertTrue(all([1 == len(v) for v in b2.pdata.values()]))
        
    
    def test_transform_remove_events(self):
        '''
        Remove events meeting some criteria from a pattern's events list
        '''
        leaf = Zound[self._pattern_id]
        l2 = self.make_leaf_pattern(4, 'fid2',store = False)
        
        
        b1 = Zound(source = 'Test')
        b1.append(leaf,[Event(i) for i in range(4)])
        b1.append(l2,[Event(i) for i in range(4)])
        
        def multiples_of_two(pattern,events):
            return pattern,filter(lambda e : e.time % 2,events)
        
        t = IndiscriminateTransform(multiples_of_two)
        b2 = b1.transform(t)
        self.assertFalse(b2 is b1)
        self.assertFalse(b2 == b1)
        self.assertTrue(leaf._id in b2.all_ids)
        self.assertTrue(l2._id in b2.all_ids)
        self.assertEqual(2,len(b2.pdata[leaf._id]))
        self.assertEqual(2,len(b2.pdata[l2._id]))
        
        
    
    def test_transform_add_events(self):
        '''
        Add events whose parameters depend on existing events
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        def double(pattern,events):
            ne = []
            for e in events:
                ne.extend([e,e >> .5])
            return pattern,ne
        
        t = IndiscriminateTransform(double)
        
        b2 = b1.transform(t)
        
        self.assertFalse(b2 is b1)
        self.assertFalse(b2 == b1)
        self.assertEqual(1,len(b2.pdata))
        self.assertEqual(8,len(b2.pdata[leaf._id]))
        
        expected = [0.5 * i for i in range(8)]
        self.assertEqual(set(expected),set([e.time for e in b2.pdata[leaf._id]]))
    
    def test_transform_nested_all_but_leaf(self):
        '''
        Alter a pattern at multiple levels of nesting, except for the leaf patterns
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = Zound(source = 'Test')
        b2.append(b1,[Event(i) for i in range(0,16,4)])
        
        # move each event forward .1 seconds in time
        def transform(pattern,events):
            
            if not events:
                return pattern,events
            
            return pattern,[e >> .1 for e in events]
        
        t = RecursiveTransform(transform)
        
        b3 = b2.transform(t)
        
        # check that things are as expected at the top level
        self.assertFalse(b3 is b2)
        self.assertFalse(b3 == b2)
        self.assertEqual(1,len(b3.pdata))
        self.assertEqual(2,len(b3.all_ids))
        self.assertTrue(leaf._id in b3.all_ids)
        self.assertFalse(b1._id in b3.all_ids)
        
        n1_id = b3.pdata.keys()[0]
        expected = set([i + .1 for i in range(0,16,4)])
        actual = set([e.time for e in b3.pdata[n1_id]])
        self.assertEqual(expected,actual)
        
        n1 = b3.patterns[n1_id]
        self.assertFalse(n1 is b1)
        self.assertFalse(n1 == b1)
        self.assertEqual(1,len(n1.pdata))
        self.assertEqual(1,len(n1.all_ids))
        self.assertTrue(leaf._id in n1.all_ids)
    

    @unittest2.expectedFailure
    def test_transform_shorten_single_leaf(self):
        '''
        Alter the last occurrence of b1 so the first leaf is shorter than the 
        others.  The first three occurrences should be unchanged.
        
        b2 --------------------
        | b1   b1   b1   b1   |   
        | xxx-|xxx-|xxx-|xxx- |
        -----------------------
        
        -->
        
        n2 --------------------
        | b1   b1   b1   n1   |   
        | xxx-|xxx-|xxx-|sxx- |
        -----------------------
        '''
        
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test',_id = 'b1')
        b1.append(leaf,[Event(i) for i in range(3)])
        
        b2 = Zound(source = 'Test',_id = 'b2')
        b2.append(b1,[Event(i) for i in range(0,16,4)])
        
        def s(pattern,events):
            # transform new leaf
            if None is events:
                addr = pattern.address
                sl = slice(addr._index.start,addr._index.stop - 2)
                new_addr = self.env.address_class((addr._id,sl))
                pattern.address = new_addr
                return pattern,events
            
            # transform leaf
            if pattern.is_leaf:
                n = pattern.copy()
                return [n,pattern],[events[0],events[1:]]
            
            # transform b1
            n = pattern.copy()
            return [pattern,n],[events[:-1],events[-1:]]
            
        t = RecursiveTransform(s,lambda p,e : \
                               (p._id == b1._id) or
                               (e and p._id == leaf._id and b1 not in p._ancestors) or
                               # BUG: all of this will be true for both the new
                               # copy and the "old" leaf, since a copy is always
                               # made
                               (p.is_leaf and p._leaf_compare(leaf) and e is None))
        
        root = b2.transform(t)
        print root.all_ids
        self.assertFalse(root is b2)
        self.assertFalse(root == b2)
        self.assertEqual(4,len(root.all_ids))
        self.assertTrue(b1._id in root.all_ids)
        self.assertTrue(leaf._id in root.all_ids)
        self.assertEqual(2,len(root.pdata))
            
    
    def test_transform_change_single_instance(self):
        '''
        Alter the last occurrence of b1 so it only has 3 beats. The first
        three occurrences should be unchanged.
        
        b2 --------------------
        | b1   b1   b1   b1   |   
        | xxxx|xxxx|xxxx|xxxx |
        -----------------------
        
        -->
        
        n2 --------------------
        | b1   b1   b1   n1   |   
        | xxxx|xxxx|xxxx|xxx- |
        -----------------------
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test',_id = 'b1')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = Zound(source = 'Test',_id = 'b2')
        b2.append(b1,[Event(i) for i in range(0,16,4)])
        
        def s(pattern,events):
            
            # make a copy of the pattern with the last beat omitted
            last = pattern.copy()
            last.pdata[leaf._id] = last.pdata[leaf._id][:-1]
             
            # b1 should play thrice, and then the new pattern should play,
            # which only has beats on [0,1,2]
            return [pattern,last],[events[:-1],events[-1:]]
        
        t = RecursiveTransform(s, lambda p,e: p._id == b1._id)
        b3 = b2.transform(t) 
        self.assertFalse(b3 is b2)
        self.assertFalse(b3 == b2)
        self.assertTrue(b1._id in b3.pdata)
        self.assertEqual(2,len(b3.pdata))
        self.assertEqual(3,len(b3.pdata[b1._id]))
    
    def test_alter_one_of_two_leaves(self):
        '''
        Alter one of two nested leaf patterns, ensuring that the containing 
        pattern is altered. Shorten only l1.
        
        b2-----------------
        | b1------------- |
        | | l1,l2,l1,l2 | |
        | --------------- |
        -------------------
        
        -->
        
        n2-----------------
        | n1------------- |
        | | n0,l2,n0,l2 | |
        | --------------- |
        -------------------
        '''
        
        l1 = self.make_leaf_pattern(1,'l1',store = False)
        l2 = self.make_leaf_pattern(2,'l2',store = False)
        
        b1 = Zound(source = 'Test', _id = 'b1')
        b1.append(l1,[Event(0),Event(2)])
        b1.append(l2,[Event(1),Event(3)])
        
        b2 = Zound(source = 'Test',_id = 'b2')
        b2.append(b1,[Event(0)])
        
        def s(pattern,events):
            # shorten the leaf pattern
            addr = pattern.address
            sl = slice(addr._index.start,addr._index.stop - 2)
            new_addr = self.env.address_class((addr._id,sl))
            pattern.address = new_addr
            return pattern,events
        
        t = RecursiveTransform(s,lambda p,e : \
                               p.is_leaf and p._leaf_compare(l1) and e is None)
        
        root = b2.transform(t)
        self.assertFalse(root is b2)
        self.assertFalse(root == b2)
        self.assertFalse(l1._id in root.all_ids)
        self.assertTrue(l2._id in root.all_ids)
        self.assertFalse(b1._id in root.all_ids)
    
    def test_replace_all_leaf_instances_with_another_leaf(self):
        '''
        Replace all x leaves with y leaves
        
        b4----------------------
        |  b2----------b3----- |
        |  | b1   b1  | b1   | |
        |  | xxxx|xxxx| xxxx | |
        |   ------------------ | 
        ------------------------
        
        -->
        
        n6----------------------
        |  n4----------n3----- |
        |  | n1   n1  | n2   | |
        |  | yyyy|yyyy| yyyy | |
        |   ------------------ | 
        ------------------------
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test',_id = 'b1')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = Zound(source = 'Test',_id = 'b2')
        b2.append(b1,[Event(i) for i in range(0,8,4)])
        
        b3 = Zound(source = 'Test', _id = 'b3')
        b3.append(b1,[Event(0)])
        
        b4 = Zound(source = 'Test', _id = 'b4')
        b4.append(b2,[Event(0)])
        b4.append(b3,Event(8))
        
        new_leaf = self.make_leaf_pattern(1,'new_leaf',store = False)
        
        def s(pattern,events):
            return new_leaf,None
        
        t = RecursiveTransform(s,lambda p,e : \
                            p.is_leaf and p._leaf_compare(leaf) and e is None)
        
        root = b4.transform(t)
        
        self.assertFalse(root is b4)
        self.assertFalse(root == b4)
        self.assertFalse(leaf._id in root.all_ids)
        self.assertTrue(new_leaf._id in root.all_ids)
        self.assertFalse(b1._id in root.all_ids)
        self.assertFalse(b2._id in root.all_ids)
        self.assertFalse(b3._id in root.all_ids)
        self.assertEqual(2,len(root.pdata))
        self.assertEqual(5,len(root.all_ids))
        
        keys = root.pdata.keys()
        k1 = keys.pop()
        self.assertEqual(1,len(root.patterns[k1].pdata))
        sk1 = root.patterns[k1].pdata.keys()[0]
        self.assertTrue(len(root.patterns[k1].pdata[sk1]) in [1,2])
        
        k2 = keys.pop()
        self.assertEqual(1,len(root.patterns[k2].pdata))
        sk2 = root.patterns[k2].pdata.keys()[0]
        self.assertTrue(len(root.patterns[k2].pdata[sk2]) in [1,2])
            
    def test_alter_only_leaf_patterns_in_specific_pattern(self):
        '''
        Alter only the b1 patterns that are children of b2.
        
        b4----------------------
        |  b2----------b3----- |
        |  | b1   b1  | b1   | |
        |  | xxxx|xxxx| xxxx | |
        |   ------------------ | 
        ------------------------
        
        -->
        
        n4----------------------
        |  n2----------b3----- |
        |  | n1   n1  | b1   | |
        |  | x-x-|x-x-| xxxx | |
        |   ------------------ | 
        ------------------------
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test',_id = 'b1')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = Zound(source = 'Test',_id = 'b2')
        b2.append(b1,[Event(i) for i in range(0,8,4)])
        
        b3 = Zound(source = 'Test', _id = 'b3')
        b3.append(b1,[Event(0)])
        
        b4 = Zound(source = 'Test', _id = 'b4')
        b4.append(b2,[Event(0)])
        b4.append(b3,Event(8))
        
        def s2(pattern,events):
            return pattern,events[::2]
        
        
        t = RecursiveTransform(s2,lambda p,e : \
                               p._id == leaf._id and b2 in p._ancestors)
        
        b5 = b4.transform(t)
        
        known_ids = set([leaf._id,b1._id,b2._id,b3._id,b4._id])
        
        self.assertFalse(b5 is b4)
        self.assertFalse(b5 == b4)
        self.assertEqual(5,len(b5.all_ids))
        self.assertTrue(leaf._id in b5.all_ids)
        self.assertTrue(b1._id in b5.all_ids)
        self.assertTrue(b3._id in b5.all_ids)
        self.assertFalse(b2._id in b5.all_ids)
        
        self.assertEqual(2,len(b5.pdata))
        self.assertTrue(b3._id in b5.pdata)
        keys = b5.pdata.keys()
        keys.remove(b3._id)
        newkey = keys[0]
        self.assertFalse(newkey in known_ids)
        pattern = b5.patterns[newkey]
        self.assertEqual(2,len(pattern.all_ids))
        self.assertTrue(leaf._id in pattern.all_ids)
        self.assertEqual(1,len(pattern.pdata))
        self.assertEqual(2,len(pattern.pdata.values()[0]))
        
        nested = pattern.pdata.keys()[0]
        self.assertFalse(nested in known_ids)
        nested = b5.patterns[nested]
        expected = set([0,2])
        actual = set([e.time for e in nested.pdata.values()[0]])
        self.assertEqual(expected,actual)
        self.assertEqual(1,len(nested.pdata))
        self.assertEqual(leaf._id,nested.pdata.keys()[0])
        
         
    
    def test_transform_nested_all(self):
        '''
        Alter a pattern at multiple levels of nesting, including the leaf patterns
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = Zound(source = 'Test')
        b2.append(b1,[Event(i) for i in range(0,16,4)])
        
        # move each event forward .1 seconds in time. Change the length of
        # leaf patterns
        def transform(pattern,events):
            if not events:
                # this is a leaf pattern
                addr = pattern.address
                sl = slice(addr._index.start,addr._index.stop - 2)
                new_addr = self.env.address_class((addr._id,sl))
                pattern.address = new_addr
                return pattern,events
            
            # this is a branch pattern
            return pattern,[e >> .1 for e in events]
        
        t = RecursiveTransform(transform)
        
        b3 = b2.transform(t)
        
        # check that things are as expected at the top level
        self.assertFalse(b3 is b2)
        self.assertFalse(b3 == b2)
        self.assertEqual(1,len(b3.pdata))
        self.assertEqual(2,len(b3.all_ids))
        self.assertFalse(leaf._id in b3.all_ids)
        self.assertFalse(b1._id in b3.all_ids)
        n1_id = b3.pdata.keys()[0]
        expected = set([i + .1 for i in range(0,16,4)])
        actual = set([e.time for e in b3.pdata[n1_id]])
        self.assertEqual(expected,actual)
        
        n1 = b3.patterns[n1_id]
        self.assertFalse(n1 is b1)
        self.assertFalse(n1 == b1)
        self.assertEqual(1,len(n1.pdata))
        self.assertEqual(1,len(n1.all_ids))
        self.assertFalse(leaf._id in n1.all_ids)
        
        n2_id = n1.pdata.keys()[0]
        n2 = n1.patterns[n2_id]
        self.assertFalse(n2 is leaf)
        self.assertFalse(n2 == leaf)
        self.assertFalse(n2.address == leaf.address)
        
    def test_transform_nested_all_two_patterns(self):
        leaf = Zound[self._pattern_id]
        p1 = Zound(source = 'Test')
        p1.append(leaf,[Event(i) for i in range(4)])
        
        p2 = Zound(source = 'Test')
        p2.append(leaf,[Event(i) for i in range(10,12)])
        
        root = Zound(source = 'Test')
        events = [Event(i) for i in [0,4]]
        root.extend([p1,p2],[events,events])
        
        # move each event forward .1 seconds in time
        def transform(pattern,events):
            
            if not events:
                return pattern,events
            
            return pattern,[e >> .1 for e in events]
        
        t = RecursiveTransform(transform)
        
        r2 = root.transform(t)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertTrue(2,len(r2.pdata))
        self.assertTrue(3,len(r2.all_ids))
        
        keys = r2.pdata.keys()
        self.assertFalse(p1._id in keys)
        self.assertFalse(p2._id in keys)
        
        self.assertTrue(all(2 == len(v) for v in r2.pdata.values()))
        patterns = [r2.patterns[k] for k in keys]
        self.assertFalse(p1 in patterns)
        self.assertFalse(p2 in patterns)
        
        self.assertTrue(all([1 == len(p.pdata) for p in patterns]))
        
        expected = [2,4]
        events = patterns[0].pdata[leaf._id]
        self.assertTrue(len(events) in expected)
        
        expected.remove(len(events))
        
        events = patterns[1].pdata[leaf._id]
        self.assertTrue(len(events) in expected)
    
    def test_transform_drill_down_change_branch(self):
        '''
        Just like test_transform_drill_down_change_leaf(), except that p2's
        events are altered instead of altering the leaf pattern l2.
        '''
        l1 = self.make_leaf_pattern(1, 'l1', store = False)
        l2 = self.make_leaf_pattern(2, 'l2', store = False)
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(l1,[Event(i) for i in range(4)])
        
        p2 = Zound(source = 'Test',_id = 'p2')
        p2.append(l2,[Event(i) for i in range(3)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(0)])
        root.append(p2,[Event(1)])
        
        def alter(pattern,events):
            return pattern,[e >> 1 for e in events]
        
        t = RecursiveTransform(alter,predicate = lambda p,e : p._id == p2._id)
        
        r2 = root.transform(t)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        
        self.assertTrue(p1._id in r2.all_ids)
        self.assertTrue(p2._id in r2.all_ids)
        self.assertTrue(l1._id in r2.all_ids)
        self.assertTrue(l2._id in r2.all_ids)
        
        self.assertEqual(2,len(r2.pdata))
        self.assertEqual(1,len(r2.pdata[p1._id]))
        self.assertEqual(4,len(r2.patterns[p1._id].pdata[l1._id]))
        self.assertEqual(1,len(r2.pdata[p2._id]))
        
        events = r2.patterns[p2._id].pdata[l2._id]
        self.assertEqual(3,len(events))
        expected = set([0,1,2])
        self.assertEqual(expected,set([e.time for e in events])) 
        
        events = r2.pdata[p2._id]
        self.assertEqual(1,len(events))
        expected = set([2])
        self.assertEqual(expected,set([e.time for e in events]))
        
        
    
    def test_transform_drill_down_change_leaf(self):
        '''
        Apply a transform that searches recursively for patterns meeeting some
        criteria.  Unmodified branches are not copied.
        '''
        
        # BUG: 
        # This will copy branches that don't change.  The transform needs to 
        # search all the way out to the leaves, but may not find patterns fitting
        # criteria anywhere along a certain branch.
        l1 = self.make_leaf_pattern(1, 'l1', store = False)
        l2 = self.make_leaf_pattern(2, 'l2', store = False)
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(l1,[Event(i) for i in range(4)])
        
        p2 = Zound(source = 'Test',_id = 'p2')
        p2.append(l2,[Event(i) for i in range(3)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(0)])
        root.append(p2,[Event(1)])
        
        def alter(pattern,events):
            addr = pattern.address
            sl = slice(addr._index.start,addr._index.stop - 2)
            new_addr = self.env.address_class((addr._id,sl))
            pattern.address = new_addr
            return pattern,events
        
        t = RecursiveTransform(alter,predicate = lambda p,e : e is None and l2._leaf_compare(p))
        r2 = root.transform(t)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        
        # this branch didn't change
        self.assertTrue(l1._id in r2.patterns)
        self.assertTrue(p1._id in r2.pdata)
        self.assertTrue(l1._id in r2.all_ids)
        self.assertTrue(p1._id in r2.all_ids)
        # this branch did
        self.assertFalse(l2._id in r2.patterns)
        self.assertFalse(p2._id in r2.pdata)
        self.assertFalse(l2._id in r2.all_ids)
        self.assertFalse(p2._id in r2.all_ids)
        
        self.assertEqual(2,len(r2.pdata))
        self.assertEqual(4,len(r2.all_ids))
        self.assertEqual(1,len(r2.pdata[p1._id]))
        p = r2.patterns[p1._id]
        self.assertEqual(1,len(p.pdata))
        self.assertEqual(4,len(p.pdata[l1._id]))
        
        
        keys = r2.pdata.keys()
        keys.remove(p1._id)
        k = keys[0]
        
        p = r2.patterns[k]
        self.assertEqual(1,len(p.pdata))
        self.assertEqual(3,len(p.pdata.values()[0]))
    
    ## AND ###########################################################
    def test_and(self):
        l1 = Zound[self._pattern_id]
        l2 = self.make_leaf_pattern(3, 'fid2', store = False)
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(l1,[Event(i) for i in range(5)])
        
        p2 = Zound(source = 'Test',_id = 'p2')
        p2.append(l2,[Event(i) for i in range(4)])
        
        r1 = Zound(source = 'Test',_id = 'r1')
        r1.append(p1,[Event(i) for i in range(7)])
        
        r2 = Zound(source = 'Test',_id = 'r2')
        r2.append(p2,[Event(i) for i in range(8)]) 
        
        p3 = r1 & r2
        
        self.assertFalse(p3 is r2)
        self.assertFalse(p3 is r1)
        self.assertFalse(p3 == r1)
        self.assertFalse(p3 == r2)
        
        self.assertEqual(2,len(p3.pdata))
        self.assertEqual(4,len(p3.all_ids))
        self.assertTrue(l1._id in p3.all_ids)
        self.assertTrue(l2._id in p3.all_ids)
        self.assertTrue(p2._id in p3.all_ids)
        self.assertTrue(p1._id in p3.all_ids)
    
    ## SHIFT #########################################################
    def test_shift_leaf(self):
        leaf = Zound[self._pattern_id]
        self.assertRaises(Exception,lambda : leaf.shift(1))
        
    def test_shift(self):
        leaf = Zound[self._pattern_id]
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(leaf,[Event(i) for i in range(4)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(i) for i in range(0,16,4)])
        
        r2 = root.shift(1)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertEqual(1,len(r2.pdata))
        
        expected = range(1,17,4)
        actual = [e.time for e in r2.pdata.values()[0]]
        self.assertEqual(expected,actual)
        
        expected = range(4)
        p = r2.patterns[p1._id]
        actual = [e.time for e in p.pdata.values()[0]]
        self.assertEqual(expected,actual)
    
    def test_shift_recursive(self):
        leaf = Zound[self._pattern_id]
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(leaf,[Event(i) for i in range(4)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(i) for i in range(0,16,4)])
        
        r2 = root.shift(1,recurse = True)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertEqual(1,len(r2.pdata))
        
        expected = range(1,17,4)
        actual = [e.time for e in r2.pdata.values()[0]]
        self.assertEqual(expected,actual)
        
        expected = range(1,5)
        p = r2.patterns[r2.pdata.keys()[0]]
        actual = [e.time for e in p.pdata.values()[0]]
        self.assertEqual(expected,actual)
    
    def test_lshift(self):
        leaf = Zound[self._pattern_id]
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(leaf,[Event(i) for i in range(1,5)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(i) for i in range(1,17,4)])
        
        r2 = root << 1
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertEqual(1,len(r2.pdata))
        
        expected = range(0,16,4)
        actual = [e.time for e in r2.pdata.values()[0]]
        self.assertEqual(expected,actual)
        
        expected = range(1,5)
        p = r2.patterns[p1._id]
        actual = [e.time for e in p.pdata.values()[0]]
        self.assertEqual(expected,actual)
    
    def test_rshift(self):
        leaf = Zound[self._pattern_id]
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(leaf,[Event(i) for i in range(4)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(i) for i in range(0,16,4)])
        
        r2 = root >> 1
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertEqual(1,len(r2.pdata))
        
        expected = range(1,17,4)
        actual = [e.time for e in r2.pdata.values()[0]]
        self.assertEqual(expected,actual)
        
        expected = range(4)
        p = r2.patterns[p1._id]
        actual = [e.time for e in p.pdata.values()[0]]
        self.assertEqual(expected,actual)
    
    ## DILATE ##############################################
    def test_dilate_leaf(self):
        leaf = Zound[self._pattern_id]
        self.assertRaises(Exception,lambda : leaf.dilate(2))
    
    def test_dilate_recursive(self):
        leaf = Zound[self._pattern_id]
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(leaf,[Event(i) for i in range(4)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(i) for i in range(0,16,4)])
        
        r2 = root.dilate(.5)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertFalse(p1._id in r2.all_ids)
        events = r2.pdata.values()[0]
        
        expected = range(0,8,2)
        actual = [e.time for e in events]
        self.assertEqual(expected,actual)
        
        _id = r2.pdata.keys()[0]
        p = r2.patterns[_id]
        
        events = p.pdata.values()[0]
        expected = [0,.5,1,1.5]
        actual = [e.time for e in events]
        self.assertEqual(expected,actual)
        
    def test_dilate(self):
        leaf = Zound[self._pattern_id]
        
        p1 = Zound(source = 'Test',_id = 'p1')
        p1.append(leaf,[Event(i) for i in range(4)])
        
        root = Zound(source = 'Test',_id = 'root')
        root.append(p1,[Event(i) for i in range(0,16,4)])
        
        r2 = root.dilate(.5,recurse = False)
        
        self.assertFalse(r2 is root)
        self.assertFalse(r2 == root)
        self.assertTrue(p1._id in r2.all_ids)
        events = r2.pdata.values()[0]
        
        expected = range(0,8,2)
        actual = [e.time for e in events]
        self.assertEqual(expected,actual)
        
        _id = r2.pdata.keys()[0]
        p = r2.patterns[_id]
        
        events = p.pdata.values()[0]
        expected = range(4)
        actual = [e.time for e in events]
        self.assertEqual(expected,actual)
    
    ## LENGTH_SAMPLES ##################################################
    
    def assert_approx_equal(self,a,b,tolerance = 1):
        self.assertTrue(abs(a-b) <= tolerance)
        
    def expected_samples(self,seconds):
        frames = self.env.seconds_to_frames(seconds)
        return self.env.frames_to_samples(frames)
    
    def test_length_samples_leaf(self):
        leaf = self.make_leaf_pattern(3, 'fid', store = False)
        self.assertEqual(self.expected_samples(3),leaf.length_samples())
    
    
    def test_length_samples_nested(self):
        leaf = self.make_leaf_pattern(3, 'fid', store = False)
        root = Zound(source = 'Test',_id = 'root')
        root.append(leaf,[Event(i) for i in range(4)])
        # the last event starts at 3 seconds, and lasts three seconds
        self.assert_approx_equal(self.expected_samples(3 + 3),
                                 root.length_samples(),
                                 tolerance = AudioConfig.windowsize)
    
    ## RENDER ############################################################
    
    def test_render_empty(self):
        p = Zound(source = 'Test')
        self.assertRaises(Exception, lambda : p._render())
    
    def test_render_leaf(self):
        leaf = self.make_leaf_pattern(2, 'fid', store = False)
        audio = leaf._render()
        
        es = self.expected_samples(2)
        self.assertEqual(es,audio.size)
        # this is a smoke test to make sure something was written
        self.assertTrue(np.abs(audio).sum() > 0)
    
    def test_render_nested(self):
        leaf = self.make_leaf_pattern(2, 'fid', store = False)
        root = Zound(source = 'Test',_id = 'root')
        root.append(leaf,[Event(i) for i in range(4)])
        audio = root._render()
        # the last event starts at 3 seconds, and lasts two seconds
        self.assert_approx_equal(self.expected_samples(3 + 2),
                                 len(audio),
                                 tolerance = AudioConfig.windowsize)
        # this is a smoke test to make sure something was written
        self.assertTrue(np.abs(audio).sum() > 0)
        
    ## AUDIO_EXTRACTOR ###################################################
    
    def test_analyze_pattern(self):
        leaf = Zound[self._pattern_id]
        root = Zound(source = 'Test',_id = 'root',external_id = 'eid')
        root.append(leaf,[Event(i) for i in range(4)])
        root.store()
        
        ec = FrameModel.extractor_chain(root)
        FrameModel.controller().append(ec)
        
        self.assertEqual(2,len(FrameModel.list_ids()))
        self.assertTrue('root' in FrameModel.list_ids())
        eid = FrameModel.controller().external_id('root')
        self.assertEqual(('Test','eid'),eid)
        
        frames = FrameModel['root']
        ls = root.length_samples()
        audio = self.env.synth(frames.audio)
        print ls
        print len(audio)
        self.assert_approx_equal(ls, len(audio), self.env.windowsize)
        
    
    ## __GETITEM__ ########################################################
    
    @unittest2.expectedFailure
    def test_get_item_pattern(self):
        self.fail()
    
    @unittest2.expectedFailure
    def test_get_item_time_slice(self):
        self.fail()
    
    ## _LEAVES_ABSOLUTE ####################################################
    
    def test_leaves_absolute_leaf(self):
        leaf = Zound[self._pattern_id]
        la = leaf._leaves_absolute()
        
        self.assertEqual(1,len(la))
        self.assertEqual(leaf._id,la.keys()[0])
        self.assertEqual(1,len(la.values()[0]))
    
    def test_leaves_absolute_nested_one_level(self):
        leaf = Zound[self._pattern_id]
        branch = Zound(source = 'Test')
        branch.append(leaf,[Event(i) for i in range(4)])
        
        la = branch._leaves_absolute()
        self.assertEqual(1,len(la))
        self.assertEqual(leaf._id,la.keys()[0])
        self.assertEqual(4,len(la.values()[0]))
    
    def test_leaves_absolute_nested_one_level_two_patterns(self):
        l1 = self.make_leaf_pattern(1, 'l1', store = False)
        l2 = self.make_leaf_pattern(2, 'l2', store = False)
        
        branch = Zound(source = 'Test')
        branch.append(l1,[Event(i) for i in range(4)])
        branch.append(l2,[Event(i) for i in range(4,8)])
        
        la = branch._leaves_absolute()
        
        self.assertEqual(2,len(la))
        self.assertTrue(l1._id in la)
        self.assertTrue(l2._id in la)
        self.assertEqual(4,len(la[l1._id]))
        self.assertEqual(4,len(la[l2._id]))
    
    def test_leaves_absolute_nested_two(self):
        leaf = Zound[self._pattern_id]
        b = Zound(source = 'Test',_id = 'branch')
        b.append(leaf,[Event(i) for i in range(4)])
        r = Zound(source = 'Test',_id = 'root')
        r.append(b,[Event(i) for i in range(0,16,4)])
        
        la = r._leaves_absolute()
        
        self.assertEqual(1,len(la))
        self.assertTrue(leaf._id in la)
        events = la[leaf._id]
        self.assertEqual(16,len(events))
    
    def test_leaves_absolute_all_unstored(self):
        l1 = self.make_leaf_pattern(1,'l1',store = False)
        l2 = self.make_leaf_pattern(2,'l2',store = False)
        b1 = Zound(source = 'Test',_id = 'b1')
        b1.append(l1,[Event(i) for i in range(4)])
        b2 = Zound(source = 'Test',_id = 'b2')
        b2.append(l2,[Event(i) for i in range(4)])
        
        combined = b1 & b2
        la = combined._leaves_absolute()
        self.assertEqual(2,len(la))
        self.assertTrue(l1._id in la)
        self.assertTrue(l2._id in la)
        self.assertEqual(4,len(la[l1._id]))
        self.assertEqual(4,len(la[l2._id]))
    
    def test_music_pattern_leaf(self):
        frames = FrameModel.random()
        leaf = MusicPattern.leaf(frames)
        self.assertTrue(isinstance(leaf,Zound))
        self.assertTrue(leaf.is_leaf)
    
    def test_music_pattern_shift_forward(self):
        l1 = self.make_leaf_pattern(1,'l1',store = False)
        l2 = self.make_leaf_pattern(2,'l2',store = False)
        mp = MusicPattern(source = 'Test',length_beats = 4,bpm = 60)
        mp.extend([l1,l2],[[Event(0),Event(1)],[Event(2),Event(3)]])
        
        shifted = mp >> 1
        t1 = [e.time for e in shifted.pdata[l1._id]]
        self.assertEqual([1,2],t1)
        
        t2 = [e.time for e in shifted.pdata[l2._id]]
        self.assertEqual([3,4],t2)
        
        flat = shifted._leaves_absolute()
        self.assertEqual(2,len(flat))
        
        # demonstrate that the events "wrapped"
        l1t = flat[l1._id]
        self.assertEqual([1,2],[e.time for e in l1t])
        
        l2t = flat[l2._id]
        self.assertEqual([3,0],[e.time for e in l2t])
    
    def test_music_pattern_shift_backward(self):
        l1 = self.make_leaf_pattern(1,'l1',store = False)
        l2 = self.make_leaf_pattern(2,'l2',store = False)
        mp = MusicPattern(source = 'Test',length_beats = 4,bpm = 60)
        mp.extend([l1,l2],[[Event(0),Event(1)],[Event(2),Event(3)]])
        
        shifted = mp << 1
        t1 = [e.time for e in shifted.pdata[l1._id]]
        self.assertEqual([-1,0],t1)
        
        t2 = [e.time for e in shifted.pdata[l2._id]]
        self.assertEqual([1,2],t2)
        
        flat = shifted._leaves_absolute()
        self.assertEqual(2,len(flat))
        
        # demonstrate that the events "wrapped"
        l1t = flat[l1._id]
        self.assertEqual([3,0],[e.time for e in l1t])
        
        l2t = flat[l2._id]
        self.assertEqual([1,2],[e.time for e in l2t])
    
    def test_music_pattern_length_samples(self):
        '''
        The value returned by length_samples() responds to changes in the 
        pattern's bpm value.
        '''
        leaf = self.make_leaf_pattern(1,'l1',store = False)
        mp = MusicPattern(source = 'Test', length_beats = 4, bpm = 60)
        mp.append(leaf,[Event(i) for i in range(4)])
        
        orig = mp.length_samples()
        self.assert_approx_equal(\
                    AudioConfig.samplerate * 4, orig, AudioConfig.windowsize)
        
        mp.bpm = 120
        faster = mp.length_samples()
        # the length at 120 bpm should be half of the original, plus the amount
        # by which the last event overflows
        expected = (orig * .5) + (.5 * AudioConfig.samplerate)
        self.assert_approx_equal(expected, faster, AudioConfig.windowsize)
    
        mp.bpm = 30
        slower = mp.length_samples()
        # the length at 30 bpm should be twice the original, minus the amount
        # by which the last event underflows
        expected = (orig * 2) - (1 * AudioConfig.samplerate)
        self.assert_approx_equal(expected, slower, AudioConfig.windowsize)
        
    def test_music_pattern_and_same_length(self):
        leaf = Zound[self._pattern_id]
        
        b1 = MusicPattern(source = 'Test', bpm = 173, length_beats = 4)
        b1.append(leaf,[Event(i) for i in range(4)])
        b2 = MusicPattern(source = 'Test', bpm = 120, length_beats = 4)
        b2.append(leaf,[Event(i) for i in range(4)])
        
        b3 = b1 & b2
        
        self.assertEqual(173,b3.bpm)
        self.assertEqual(4,b3.length_beats)
        self.assertEqual(1,len(b3.pdata))
        self.assertEqual(8,len(b3.pdata[leaf._id]))
        
    
    def test_music_pattern_different_lengths(self):
        leaf = Zound[self._pattern_id]
        
        b1 = MusicPattern(source = 'Test', bpm = 173, length_beats = 4)
        b1.append(leaf,[Event(i) for i in range(4)])
        b2 = MusicPattern(source = 'Test', bpm = 120, length_beats = 8)
        b2.append(leaf,[Event(i) for i in range(8)])
        
        b3 = b1 & b2
        
        self.assertEqual(173,b3.bpm)
        self.assertEqual(8,b3.length_beats)
        self.assertEqual(1,len(b3.pdata))
        self.assertEqual(12,len(b3.pdata[leaf._id]))
    
    def test_music_pattern_add(self):
        leaf = Zound[self._pattern_id]
        
        b1 = MusicPattern(source = 'Test', bpm = 173, length_beats = 2)
        b1.append(leaf,[Event(i) for i in range(2)])
        b2 = MusicPattern(source = 'Test', bpm = 120, length_beats = 4)
        b2.append(leaf,[Event(i) for i in range(4)])
        
        b3 = b1 + b2
        self.assertEqual(173,b3.bpm)
        self.assertEqual(6,b3.length_beats)
        self.assertEqual(1,len(b3.pdata))
        self.assertEqual(6,len(b3.pdata[leaf._id]))
    
    def test_music_pattern_add_lhs_has_negative_beats(self):
        l1 = self.make_leaf_pattern(1, 'l1', store= False)
        l2 = self.make_leaf_pattern(1, 'l2', store = False)
        
        p1 = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        p1.append(l1,[Event(i) for i in range(4)])
        p1 = p1 << 1
        
        p2 = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        p2.append(l2,[Event(i) for i in range(4)])
        
        p3 = p1 + p2
        
        la = p3._leaves_absolute()
        self.assertEqual(2,len(la))
        self.assertTrue(l1._id in la)
        self.assertTrue(l2._id in la)
        
        # ensure that the negative time in p1 was interpreted in the context
        # of that pattern, and not in the context of the new, longer pattern
        # resulting from the addition
        expected = set([0,1,2,3])
        actual = set([e.time for e in la[l1._id]])
        self.assertEqual(expected,actual)
        
        expected = set([4,5,6,7])
        actual = set([e.time for e in la[l2._id]])
        self.assertEqual(expected,actual)
    
    def test_music_pattern_sum(self):
        leaf = Zound[self._pattern_id]
        
        b1 = MusicPattern(source = 'Test', bpm = 173, length_beats = 2)
        b1.append(leaf,[Event(i) for i in range(2)])
        b2 = MusicPattern(source = 'Test', bpm = 120, length_beats = 4)
        b2.append(leaf,[Event(i) for i in range(4)])
        b3 = MusicPattern(source = 'Test', bpm = 200, length_beats = 6)
        b3.append(leaf,[Event(i) for i in range(6)])
        
        b4 = sum([b1,b2,b3])
        
        self.assertEqual(173,b4.bpm)
        self.assertEqual(12,b4.length_beats)
        self.assertEqual(1,len(b4.pdata))
        self.assertEqual(12,len(b4.pdata[leaf._id]))
    
    def test_music_pattern_multiply(self):
        leaf = Zound[self._pattern_id]
        
        b1 = MusicPattern(source = 'Test', bpm = 173, length_beats = 4)
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = b1 * 5
        
        self.assertEqual(173,b2.bpm)
        self.assertEqual(20,b2.length_beats)
        self.assertEqual(1,len(b2.pdata))
        
        key = b2.pdata.keys()[0]
        
        self.assertTrue(key != leaf._id)
        
        self.assertEqual(5,len(b2.pdata[key]))
    
    def test_music_pattern_invert_one_level(self):
        leaves = [self.make_leaf_pattern(1,'l%i'%i,store = False) for i in range(4)]
        mp = MusicPattern(source = 'Test', bpm = 60, length_beats = 4)
        mp.extend(leaves,[[Event(i)] for i in range(len(leaves))])
        
        # invert the pattern
        bw = ~mp
        
        self.assertEqual(4,len(bw.pdata))
        for i,l in enumerate(leaves):
            event = bw.pdata[l._id][0]
            self.assertEqual(-(i + 1),event.time)
        
        la = bw._leaves_absolute()
        self.assertEqual(4,len(la))
        for i,l in enumerate(leaves):
            event = la[l._id][0]
            self.assertEqual(3 - i,event.time)
    
    @unittest2.expectedFailure
    def test_music_pattern_invert_two_levels(self):
        leaves = [self.make_leaf_pattern(1,'l%i'%i,store = False) for i in range(4)]
        b1 = MusicPattern(source = 'Test', bpm = 60, length_beats = 4,_id = 'b1')
        b1.extend(leaves,[[Event(i)] for i in [0,3,2,1]])
        
        b2 = MusicPattern(source = 'Test', bpm = 60, length_beats = 4,_id = 'b2')
        b2.extend(leaves,[[Event(i)] for i in [3,2,0,1]])
        
        root = MusicPattern(source = 'Test', bpm = 60, length_beats = 8,_id = 'root')
        root.append(b1,[Event(0)])
        root.append(b2,[Event(4)])
        
        bw = ~root
        self.assertEqual(8,bw.length_beats)
        la = bw._leaves_absolute()
        self.assertEqual(4,len(la))
        
        l0 = set([e.time for e in la[leaves[0]._id]])
        self.assertEqual(set([0,7]),l0)
        
        l1 = set([e.time for e in la[leaves[1]._id]])
        self.assertEqual(set([1,4]),l1)
        
        l2 = set([e.time for e in la[leaves[2]._id]])
        self.assertEqual(set([3,5]),l2)
        
        l3 = set([e.time for e in la[leaves[3]._id]])
        self.assertEqual(set([2,6]),l3)
    
    def test_music_pattern_invert_two_levels_2(self):
        # create a backbeat pattern
        h = self.make_leaf_pattern(1,'hihat',store = False)
        s = self.make_leaf_pattern(1,'snare',store = False)
        k = self.make_leaf_pattern(1,'kick',store = False)
        
        hp = MusicPattern(source = 'Test',bpm = 60)
        hp.append(h,[Event(i) for i in np.arange(0,4,.5)])
        
        sp = MusicPattern(source = 'Test',bpm = 60)
        sp.append(s,[Event(1),Event(3)])
        
        kp = MusicPattern(source = 'Test',bpm = 60)
        kp.append(k,[Event(0),Event(2)])
        
        la = (~sp)._leaves_absolute()
        self.assertEqual([0,2],[e.time for e in la['snare']])
        
    def test_music_pattern_render(self):
        '''
        This is really just a smoke test to make sure that _render doesn't
        raise an exception
        '''
        leaf = Zound[self._pattern_id]
        pattern = MusicPattern(source = 'Test',length_beats = 4)
        pattern.append(leaf,[Event(0),Event(2)])
        audio = pattern._render()
        self.assertTrue(audio.dtype == np.float32)
    
    def test_store_pattern_from_leaf_method(self):
        
        # there's only one frames instance in the database, so we know what to
        # expect here
        
        frames = FrameModel.random()
        leaf = Zound.leaf(frames)
        leaf.store()
        
        l2 = Zound[leaf._id]
        self.assertFalse(l2 is leaf)
        self.assertTrue(l2 == leaf)
        self.assertTrue(l2.is_leaf)
    
    def test_render_nested_music_pattern(self):
        frames = FrameModel.random()
        leaf = Zound.leaf(frames)
        p1 = MusicPattern(source = 'Test')
        # eight notes
        p1.append(leaf,[Event(i) for i in [0,.5,1,1.5,2,2.5,3,3.5]])
        p1.store()
        
        # repeat p1 four times
        p2 = p1 * 4
        
        # render the new pattern
        audio = p2._render()
        
        
    def test_retrieve_stored_and_analyzed_pattern(self):
        # See note in model.pattern.Zound.fromdict(), line 1025
        frames = FrameModel.random()
        
        # create a musical pattern and store it
        leaf = Zound.leaf(frames)
        p1 = MusicPattern(source = 'Test')
        p1.append(leaf,[Event(i) for i in [0,.5,1,1.5,2,2.5,3,3.5]])
        p1.store()
        _id = p1._id
        del p1
        
        # fetch it and analyze it
        p1 = MusicPattern[_id]
        self.assertTrue(p1.address is None)
        ec = FrameModel.extractor_chain(p1)
        addr = FrameModel.controller().append(ec)
        p1.address = addr
        p1.store()
        
        del p1
        
        p1 = MusicPattern[_id]
        self.assertEqual(addr,p1.address)
        self.assertEqual(1,len(p1.pdata))
        self.assertEqual(1,len(p1.all_ids))
    
    
    ## MusicPattern.interpret_time #######################################
    
    def test_interpret_time_lt_length_beats(self):
        mp = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        self.assertEqual(1,mp.interpret_time(1))
    
    def test_interpret_time_gt_length_beats(self):
        mp = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        self.assertEqual(1,mp.interpret_time(5))
    
    def test_interpret_time_negative_lt_length_beats(self):
        mp = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        self.assertEqual(3,mp.interpret_time(-1))
    
    def test_interpret_time_negative_gt_length_beats(self):
        mp = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        self.assertEqual(3,mp.interpret_time(-5))
    
    
    def test_add_pattern_and_inverted(self):
        leaf = Zound[self._pattern_id]
        p = MusicPattern(source = 'Test',length_beats = 4, bpm = 60)
        p.append(leaf,[Event(i) for i in range(4)])
        # This should be an eight beat long pattern with an event on every whole
        # beat
        p2 = (p + (~p))
        
        expected = set(range(8))
        actual = set([e.time for e in p2._leaves_absolute()[leaf._id]])
        
        self.assertEqual(expected,actual)
    
    def test_add_lhs_leaf(self):
        l1 = self.make_leaf_pattern(1,'l1',store = False)
        l2 = self.make_leaf_pattern(2,'l2',store = False)
        p = MusicPattern(source = 'Test')
        p.append(l2,[Event(i) for i in range(4)])
        
        self.assertRaises(ValueError,lambda : l1 + p)
    
    def test_add_rhs_leaf(self):
        l1 = self.make_leaf_pattern(1,'l1',store = False)
        l2 = self.make_leaf_pattern(2,'l2',store = False)
        p = MusicPattern(source = 'Test')
        p.append(l2,[Event(i) for i in range(4)])
        
        self.assertRaises(ValueError,lambda : p + l1)
        
        
        
class InMemoryTest(unittest2.TestCase,PatternTest):
    
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


class MongoDbTest(unittest2.TestCase,PatternTest):
    
    def setUp(self):
        self._pattern_controller = MongoDbPatternController(dbname = 'zounds_test')
        try:
            self.set_up()
        except Exception as e:
            # KLUDGE: This is a stop-gap solution for when set_up fails.  Get
            # rid of this.
            print 'SETUP FAILED'
            print e
    
    def tearDown(self):
        self._pattern_controller._cleanup()
        self.tear_down()