import unittest
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
    RecursiveTransform

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
    
    def test_store_twice(self):
        '''
        create a pattern and call store() twice. The second call should raise an
        exception
        '''
        leaf = Zound[self._pattern_id]
        self.assertRaises(Exception,lambda : leaf.store())
    
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
        
        
    def test_append_stored_nested(self):
        '''
        Similar to test_append_stored(), but with two levels of nesting
        '''
        self.fail()
    
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
        
    def test_append_nested(self):
        self.fail()
        
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
    
    # TRANSFORM
    # remove leaf, alter leaf, or alter leaf's event's list
    # 
    '''
    transform() takes a dictionary-like object in the form
    {key(s) : action}
    
    key can be a wildcard, a key, or a list of keys that the action should be
    taken on
    
    action is a callable in the form
    action(pattern,events)
    
    action can alter the pattern, the events list, or both
    
    KLUDGE: What if I want to turn a single pattern with multiple events into
    many patterns, each with one event?
    '''
    
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
    
    def test_transform_change_single_instance(self):
        '''
        Alter only the last occurrence of b1 do it only has 3 beats. The first
        three occurrences should continue to have four beats.
        '''
        leaf = Zound[self._pattern_id]
        b1 = Zound(source = 'Test')
        b1.append(leaf,[Event(i) for i in range(4)])
        
        b2 = Zound(source = 'Test')
        b2.append(b1,[Event(i) for i in range(0,16,4)])
        self.fail()
    
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
        self.assertFalse(p2._id in r2.all_ids)
        self.assertTrue(l1._id in r2.all_ids)
        self.assertTrue(l2._id in r2.all_ids)
        
        self.assertEqual(2,len(r2.pdata))
        self.assertEqual(1,len(r2.pdata[p1._id]))
        self.assertEqual(4,len(r2.patterns[p1._id].pdata[l1._id]))
        
        keys = r2.pdata.keys()
        keys.remove(p1._id)
        k = keys[0]
        
        self.assertEqual(1,len(r2.pdata[k]))
        self.assertEqual(3,len(r2.patterns[k].pdata[l2._id]))
        
    
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
    
    ## ADD ###########################################################
    def test_add(self):
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
        
        p3 = r1 + r2
        
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
        
        
    
    ## SUM ###########################################################
    def test_sum(self):
        l1 = self.make_leaf_pattern(1, 'l1', store = False)
        l2 = self.make_leaf_pattern(2, 'l2', store = False)
        l3 = self.make_leaf_pattern(3, 'l3', store = False)
        
        p1 = Zound(source = 'Test', _id = 'p1')
        p1.append(l1,[Event(i) for i in range(1)])
        
        p2 = Zound(source = 'Test', _id = 'p2')
        p2.append(l2,[Event(i) for i in range(1)])
        
        p3 = Zound(source = 'Test', _id = 'p3')
        p3.append(l3,[Event(i) for i in range(1)])
        
        s = sum([p1,p2,p3])
        
        self.assertEqual(3,len(s.pdata))
        self.assertEqual(3,len(s.all_ids))
        self.assertTrue(l1._id in s.all_ids)
        self.assertTrue(l2._id in s.all_ids)
        self.assertTrue(l3._id in s.all_ids)
    
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
        self.assertEqual(self.expected_samples(3),leaf.length_samples)
    
    
    def test_length_samples_nested(self):
        leaf = self.make_leaf_pattern(3, 'fid', store = False)
        root = Zound(source = 'Test',_id = 'root')
        root.append(leaf,[Event(i) for i in range(4)])
        # the last event starts at 3 seconds, and lasts three seconds
        self.assert_approx_equal(self.expected_samples(3 + 3),
                                 root.length_samples,
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
        ls = root.length_samples
        audio = self.env.synth(frames.audio)
        self.assert_approx_equal(ls, len(audio), self.env.windowsize)
        
    
    ## __GETITEM__ ########################################################
    
    def test_get_item_pattern(self):
        self.fail()
    
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


class MongoDbTest(unittest.TestCase,PatternTest):
    
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