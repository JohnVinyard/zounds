from config import *
from acquire.acquirer import DiskAcquirer
import cProfile
import sys
if __name__ == '__main__':
    path = sys.argv[1]
    DiskAcquirer(path).acquire()
    