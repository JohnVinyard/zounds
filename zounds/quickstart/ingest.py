from config import *
from zounds.acquire.acquirer import DiskAcquirer
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa('--path',
       help = 'the directory to ingest sound files from',
       required = True)
    args = parser.parse_args()
    path = args.path
    DiskAcquirer(path).acquire()
    