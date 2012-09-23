from abc import ABCMeta, abstractmethod

from controller import Controller,PickledController


class SearchController(Controller):
    
    '''
    A controller that persists and fetches searches
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __delitem__(self):
        pass
    
    @abstractmethod
    def __getitem__(self):
        pass
    
    @abstractmethod
    def store(self,pipeline):
        pass

class PickledSearchController(PickledController,SearchController):
    '''
    A search controller that persists Searches by pickling them to disk
    '''