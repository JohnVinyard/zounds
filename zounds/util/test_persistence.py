import unittest2
from .persistence import simple_in_memory_settings


@simple_in_memory_settings
class X(object):
    pass


class PersitenceDecoratorTests(unittest2.TestCase):
    def test_in_memory_settings_maintains_class_name(self):
        self.assertEqual('X', X.__name__)
        self.assertNotEqual('zounds.util.persistence', X.__module__)
