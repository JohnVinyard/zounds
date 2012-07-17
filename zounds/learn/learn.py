from abc import ABCMeta, abstractmethod

class Learn(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def train(self,data,stopping_condition):
        '''
        Execute some sort of supervised or unsupervised learning algorithm
        :param data: raw data to train on, e.g., mfcc patches or labeled training
        examples
        :param stopping_condition: a callable that should be checked periodically
        by the train() implementation to determine when the learning is complete.
        '''
        pass
    
    @abstractmethod
    def __call__(self,data):
        '''
        Extract features from the data
        '''
        pass