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
    
    def test_shift_forward(self):
        e = Event(1)
        e2 = e.shift(.5)
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(1.5,e2.time)
    
    def test_shift_backward(self):
        e = Event(1)
        e2 = e.shift(-.5)
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(.5,e2.time)
    
    def test_rshift(self):
        e = Event(1)
        e2 = e >> .5
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(1.5,e2.time)
    
    def test_lshift(self):
        e = Event(1)
        e2 = e << .5
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(.5,e2.time)
    
    def test_multiply(self):
        e = Event(10)
        e2 = e * 5
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(50,e2.time)
    
    def test_add(self):
        e = Event(10)
        e2 = e + 2
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(12,e2.time)
    
    def test_radd(self):
        e = Event(10)
        e2 = 2 + e
        
        self.assertFalse(e2 is e)
        self.assertFalse(e2 == e)
        self.assertEqual(12,e2.time)
    
    