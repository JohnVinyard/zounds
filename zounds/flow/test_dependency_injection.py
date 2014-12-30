import unittest2
from dependency_injection import Registry,dependency

class SomeClass(object):
    pass

class DependencyInjectionTest(unittest2.TestCase):
    
    def setUp(self):
        Registry.clear()
    
    def test_can_register_singleton(self):
        o = object()
        Registry.register(SomeClass,o)
        o2 = Registry.get_instance(SomeClass)
        self.assertEqual(id(o),id(o2))
    
    def test_can_register_callable_with_only_args(self):
        def x(a,b):
            return (a,b)
        Registry.register(SomeClass, x)
        inst = Registry.get_instance(SomeClass,10,20)
        self.assertEqual((10,20),inst)
        
    def test_can_register_callable_with_only_kwargs(self):
        def x(madame = None, psychosis = None):
            return (madame,psychosis)
        Registry.register(SomeClass, x)
        inst = Registry.get_instance(SomeClass,madame = 10, psychosis = 20)
        self.assertEqual((10,20),inst)
    
    def test_can_register_callable_with_args_and_kwargs(self):
        def x(enfield,tennis_academy = None):
            return (enfield,tennis_academy)
        Registry.register(SomeClass,x)
        inst = Registry.get_instance(SomeClass,10,tennis_academy = 20)
        self.assertEqual((10,20),inst)
    
    # BUG: It's not possible to register a class that implements __call__ as a
    # singleton
    @unittest2.skip
    def test_can_register_callable_as_singleton(self):
        class X(object):
            
            def __call__(self):
                pass
        
        singleton = X()
        Registry.register(SomeClass,singleton)
        inst = Registry.get_instance(SomeClass)
        self.assertEqual(id(singleton),id(inst))
    
    def test_trying_to_get_instance_of_unregistered_dependency_throws_sane_error(self):
        self.assertRaises(KeyError,lambda : Registry.get_instance(SomeClass))
    
    def test_can_create_dependency_that_is_also_a_property(self):
        o = object()
        Registry.register(SomeClass, o)
        
        class X(object):
            @property    
            @dependency(SomeClass)
            def method(self):
                pass
        
        inst = X()
        self.assertEqual(id(o),id(inst.method))
    
    def test_dependency_retains_original_name(self):
        class X(object):
            @dependency(SomeClass)
            def method(self):
                pass
        inst = X()
        self.assertEqual('method',inst.method.__name__)
            
    