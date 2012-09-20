from abc import ABCMeta, abstractmethod
from zounds.util import tostring

class Learn(object):
    '''
    Learn is an abstract base class.  Derived classes must implement the 
    :py:meth:`Learn.train` and :py:meth:`Learn.__call__` methods.
    
    The :py:meth:`Learn.train` method should "learn" some new representation of
    training examples.
    
    Once trained, the :py:meth:`__call__` method should transform data into the
    learned representation. 
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def train(self,data,stopping_condition):
        '''
        Execute a supervised or unsupervised learning algorithm
        
        :param data: raw data to train on, e.g., mfcc patches or labeled training \
        examples
        
        :param stopping_condition: a callable that should be checked periodically \
        by the :code:`train()` implementation to determine when the learning is complete.
        '''
        pass
    
    @abstractmethod
    def __call__(self,data):
        '''__call__
        
        Extract features from the data
        
        :param data: A numpy array of data to be transformed.
        '''
        pass
    
    def __repr__(self):
        return tostring(self)
    
    def __str__(self):
        return self.__repr__()
    