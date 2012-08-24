import numpy as np
import unittest
from zounds.nputil import pad
from random import shuffle

class EnsurePathExistsTests(unittest.TestCase):
    
    def test_filepath(self):
        self.fail()
    
    def test_directory(self):
        self.fail()

class PadTests(unittest.TestCase):
    
    def test_pad_onedim_desired(self):
        a = np.array([1,2,3])
        b = pad(a,3)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
        
    def test_pad_onedim_longer(self):
        a = np.array([1,2,3,4])
        b = pad(a,3)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
        
    def test_pad_onedim_shorter(self):
        a = np.array([1,2,3])
        b = pad(a,4)
        self.assertEqual(4,len(b))
        self.assertEqual(0,b[-1])
        
    def test_pad_twodim_desired(self):
        a = np.random.random_sample((10,10))
        b = pad(a,10)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
    
    def test_pad_twodim_longer(self):
        a = np.random.random_sample((12,10))
        b = pad(a,10)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.allclose(a,b))
        
    def test_pad_twodim_shorter(self):
        a = np.random.random_sample((10,10))
        b = pad(a,13)
        self.assertEqual(13,b.shape[0])
        self.assertTrue(np.allclose(a,b[:10]))
        self.assertTrue(np.all(b[10:] == 0))
        
    def test_pad_list(self):
        l = [1,2,3]
        b = pad(l,4)
        self.assertEqual(4,len(b))
        self.assertEqual(0,b[-1])

from util import recurse,sort_by_lineage
class Node:
    
    def __init__(self,parents=None):
        if not parents:
            self.parents = []
        else:
            self.parents = parents
    
    @recurse
    def ancestors(self):
        return self.parents
    
class RecurseTests(unittest.TestCase):
    
    def test_root(self):
        n = Node()
        self.assertEqual(0,len(n.ancestors()))
        
    def test_single_parent(self):
        root = Node()
        child = Node(parents=[root])
        a = child.ancestors()
        self.assertEqual(1,len(a))
        self.assertTrue(root in a)
        
    def test_multi_ancestor(self):
        root = Node()
        c1 = Node(parents=[root])
        c2 = Node(parents=[root])
        gc1 = Node(parents=[c1,c2])
        a = gc1.ancestors()
        self.assertEqual(3,len(a))
        self.assertTrue(root in a)
        self.assertTrue(c1 in a)
        self.assertTrue(c2 in a)

class SortByLineageTests(unittest.TestCase):
    
    def test_single_parent(self):
        n1 = Node()
        n2 = Node([n1])
        l = [n2,n1]
        l.sort(sort_by_lineage(Node.ancestors))
        self.assertEqual(l[0],n1)
        self.assertEqual(l[1],n2)
        
    def test_multi_ancestor(self):
        n1 = Node()
        n2 = Node()
        n3 = Node([n1,n2])
        l = [n3,n2,n1]
        l.sort(sort_by_lineage(Node.ancestors))
        self.assertEqual(l[-1],n3)
    
    def test_complex(self):
        _id = Node()
        source = Node()
        framen = Node()
        external_id = Node()
        fft = Node()
        loudness = Node([fft])
        centroid = Node([fft])
        flatness = Node([fft])
        bark = Node([fft])
        rbm = Node([bark])
        l = [_id,source,framen,external_id,fft,loudness,centroid,flatness,bark,rbm]
        shuffle(l)
        l.sort(sort_by_lineage(Node.ancestors))
        self.assertTrue(l.index(rbm) > l.index(bark))
        
        