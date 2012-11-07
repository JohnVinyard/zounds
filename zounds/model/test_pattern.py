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
    
    