import unittest
from random import random
import numpy as np

from zounds.model.pattern import Event


class EventTests(unittest.TestCase):
    
    def test_sort(self):
        events = [Event(random()) for i in range(10)]
        events.sort()
        # if the list is sorted, the times will always increase, so the values
        # of diff will always be positive
        diff = np.diff([e.time for e in events])
        self.assertTrue(np.all(diff > 0))
    
    def assert_expected(self,e,ne,value):
        self.assertFalse(ne is e)
        self.assertFalse(ne == e)
        self.assertEqual(value,ne.time)
        
    def test_shift_forward(self):
        e = Event(1)
        e2 = e.shift(.5)
        self.assert_expected(e,e2,1.5)
    
    def test_shift_backward(self):
        e = Event(1)
        e2 = e.shift(-.5)
        self.assert_expected(e,e2,.5)
    
    def test_rshift(self):
        e = Event(1)
        e2 = e >> .5
        self.assert_expected(e, e2, 1.5)
    
    def test_lshift(self):
        e = Event(1)
        e2 = e << .5
        self.assert_expected(e,e2,.5)
    
    def test_multiply(self):
        e = Event(10)
        e2 = e * 5
        self.assert_expected(e,e2,50)
    
    def test_add(self):
        e = Event(10)
        e2 = e + 2
        self.assert_expected(e,e2,12)
    
    def test_radd(self):
        e = Event(10)
        e2 = 2 + e
        self.assert_expected(e,e2,12)
    
    def test_sub(self):
        e = Event(10)
        e2 = e - 2
        self.assert_expected(e,e2,8)
    
    def test_rsub(self):
        e = Event(2)
        e2 = 10 - e
        self.assert_expected(e,e2,8)
    
    