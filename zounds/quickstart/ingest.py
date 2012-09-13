from config import *
from zounds.acquire.acquirer import DiskAcquirer
import argparse
import os

LOGGER = logging.getLogger('zounds.ingest')

def fetch_sounds():
    import urllib
    import tarfile
    # KLUDGE: How can I avoid hard-coding the url here?
    url = 'http://www.johnvinyard.com/zoundsdoc/zounds_quickstart_sounds.tar.gz'
    # the name of the file to save the sound archive to
    filename = 'sounds.tar.gz'
    # get the archive
    LOGGER.info('Downloading a few sounds from %s...',url)
    urllib.urlretrieve(url, filename)
    # extract everything
    LOGGER.info('Extracting the sounds...')
    tf = tarfile.open(filename)
    # KLUDGE: This assumes that the archive was created from a directory, and that
    # there is only a single directory containing files.  Also, will the root
    # directory always be the first directory in the the getmembers() list?
    sounddir = filter(lambda m : m.isdir(),tf.getmembers())[0].name
    # flatten the archive. place all audio files from the archive into sounddir
    tf.extractall()
    # remove the archive file
    LOGGER.info('Cleaning up...')
    os.remove(filename)
    return sounddir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa('--path',
       help = 'the directory to ingest sound files from',
       required = True)
    args = parser.parse_args()
    if args.path:
        path = args.path
    else:
        path = fetch_sounds()
        
    DiskAcquirer(path).acquire()
    