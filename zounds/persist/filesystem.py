from zounds.flow.data import Database
from zounds.util import ensure_path_exists
import os

class FileSystemDatabase(Database):
    
    def __init__(self,path = 'data'):
        super(FileSystemDatabase,self).__init__()
        self._path = path
        ensure_path_exists(self._path) 
    
    def write_stream(self,key,content_type):
        return open(os.path.join(self._path,key),'wb')
    
    def read_stream(self,key):
        return open(os.path.join(self._path,key),'rb')