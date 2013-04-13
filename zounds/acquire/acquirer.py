from __future__ import division
from abc import ABCMeta,abstractmethod,abstractproperty
import os.path
from time import time
from multiprocessing import Manager,Pool
import urlparse
import traceback
import logging

from zounds.constants import available_file_formats
from zounds.environment import Environment
from zounds.model.pattern import FilePattern,UrlPattern

LOGGER = logging.getLogger(__name__)

def audio_files(path):
    '''
    Return the name of each sound file that Zounds can process in
    the given directory
    '''
    # list all files in the directory
    allfiles = os.listdir(path) 
    return filter(\
        lambda f : os.path.splitext(f)[1][1:] in available_file_formats,
        allfiles)


class Acquirer(object):
    '''
    Acquirers fetch sound files from disk, or remote locations (e.g., 
    Freesound.org) and enqueue them for processing and insertion into the
    database
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
        self.env = Environment.instance
    
    @property
    def framemodel(self):
        '''
        The :py:class:`~zounds.model.frame.Frames`-derived class defined for the
        current environment
        '''
        return self.env.framemodel
    
    @property
    def framecontroller(self):
        '''
        The :py:class:`~zounds.data.frame.frame.FrameController`-derived instance that
        the current enviornment is configured to use.
        '''
        return self.framemodel.controller()
    
    def extractor_chain(self,pattern):
        return self.framemodel.extractor_chain(pattern)
    
    @abstractproperty
    def source(self):
        '''
        **Must be implemented by inheriting classes**
        
        The source name, e.g. 'Freesound' or 'StemsFromMyAlbum'
        
        A string which identifies the source of the audio files to be consumed
        by this acquirer.  The identifier may be constant, or may vary depending
        on parameters used when constructing this :py:class:`Acquirer` instance.
        '''
        pass
    
    @abstractmethod
    def _acquire(self):
        '''
        **Must be implemented by inheriting classes**
        
        Do the work necessary to fetch audio, analyze it, and insert it into the
        database
        '''
        pass
    
    def acquire(self):
        '''
        Do the work necessary to acquire sound data, process it, and
        insert it into the db. Ensure that the database's indexes are updated
        when all data has been ingested.
        '''
        # let subclasses do the dirty work
        self._acquire()
        # update indexes, if applicable for this data store
        self.framecontroller.update_index()


def acquire_multi(args):
    env_args,path,source,files,filen,total_files = args
    Z = Environment.reanimate(env_args)
    total_frames = 0
    for i,fn in enumerate(files):
        fp = os.path.join(path,fn)
        extid = os.path.splitext(fn)[0]
        pattern = FilePattern(Z.newid(),source,extid,fp)
        if not Z.framecontroller.exists(source,extid):
            try:
                LOGGER.info(DiskAcquirer.processing_message(\
                                            source, fn, filen + i, total_files))
                addr = Z.framecontroller.append(Z.framemodel.extractor_chain(pattern))
                total_frames += len(addr)
            except:
                # KLUDGE: Do some real logging here
                # TODO: How do I recover from an error once partial data has
                # been written? 
                LOGGER.exception(traceback.format_exc())
        else:
            LOGGER.info(DiskAcquirer.skip_message(source, fn))
    return total_frames

# TODO: Handle web-services, like FreeSound, or SoundCloud
class UrlAcquirer(Acquirer):
    '''
    Import one or more audio files, given an iterable of URIs
    '''
    
    def __init__(self,baseurl,paths):
        Acquirer.__init__(self)
        self.baseurl = baseurl
        self.paths = paths
    
    @property
    def source(self):
        return urlparse.urlparse(self.baseurl).hostname
    
    def _acquire(self):
        # TODO: Enable multiprocessing, just like in DiskAcquirer
        frames_processed = 0
        lf = len(self.paths)
        for i,path in enumerate(self.paths):
            uri = urlparse.urljoin(self.baseurl,path)
            extid = path
            fn = uri
            pattern = UrlPattern(self.env.newid(),self.source,path,uri)
            
            # TODO: Factor the following into the base class.  It's
            # identical to DiskAcquirer
            if not self.framecontroller.exists(self.source,extid):
                try:
                    LOGGER.info(DiskAcquirer.processing_message(self.source, fn, i, lf))
                    addr = \
                        self.framecontroller.append(self.extractor_chain(pattern))
                    frames_processed += len(addr)
                except:
                    # KLUDGE: Do some real logging here
                    # TODO: How do I recover from an error once partial data has
                    # been written?
                    LOGGER.exception(traceback.format_exc())
            else:
                LOGGER.info(DiskAcquirer.skip_message(self.source, fn))
            
            
    
class DiskAcquirer(Acquirer):
    '''
    Import all audio files from a single directory into the zounds database.  
    Note that mp3 files are not currently supported, and will be skipped.
    '''
    
    def __init__(self,path,source = None):
        '''__init__
        
        :param path: path to a directory containing one or more audio files. All \
        files with supported formats (almost everything except mp3) will be \
        analyzed and stored when :py:meth:`Acquirer.acquire` is called.
        
        :param source: If :code:`None`, the :py:meth:`Acquirer.source` will \
        default to the final segment of :code:`path`.  If provided, \
        :py:meth:`Acquirer.source` will return this value.
        
        '''
        Acquirer.__init__(self)
        self.path = os.path.normpath(path)
        self._source = source if source else os.path.split(self.path)[1]
        parallel = \
            Environment.parallel() and self.framecontroller.concurrent_writes_ok
        self.__acquire = \
            self._acquire_multi if parallel else self._acquire_single
        
    @property
    def source(self):
        return self._source
    
    @staticmethod
    def processing_message(source,external_id,filen,total_files):
        return 'importing %s, %s, file %i of %i' % \
                (source,external_id,filen + 1,total_files)
    
    @staticmethod
    def skip_message(source,external_id):
        return 'skipping %s, %s. It\'s already in the database.' % \
                (source,external_id)
    
    @staticmethod
    def complete_message(total_seconds,seconds_spent_working):
        return 'Processed %1.4f seconds of audio in %1.4f seconds. %1.4f%% realtime.' % \
            (total_seconds,seconds_spent_working,seconds_spent_working / total_seconds)
    
    @staticmethod
    def no_files_processed_message(source):
        return '%s was empty, or all the files contained therein have been processed already.' \
                 % source
    
    def _acquire(self):
        files = audio_files(self.path)
        start = time()
        seconds_processed = self.__acquire(files)
        elapsed = time() - start
        # TODO: Is the following line necessary?  Isn't this called in the base
        # acquire() method?
        self.framecontroller.update_index()
        if not seconds_processed:
            LOGGER.info(DiskAcquirer.no_files_processed_message(self._source))
        else:
            LOGGER.info(DiskAcquirer.complete_message(seconds_processed, elapsed)) 
    
    def _acquire_multi(self,files):
        mgr = Manager()
        lock = mgr.Lock()
        lf = len(files)
        args = []
        total_frames = 0
        chunksize = 15
        env_args = self.env.__getstate__(lock = lock)
        for i in range(0,len(files),chunksize):
            filechunk = files[i : i + chunksize]
            args.append((env_args,self.path,self._source,filechunk,i,lf))
        p = Pool(Environment.n_cores)
        total_frames += sum(p.map(acquire_multi,args))
        return self.env.frames_to_seconds(total_frames)
        
        
    def _acquire_single(self,files):
        lf = len(files)
        frames_processed = 0
        for i,fn in enumerate(files):
            fp = os.path.join(self.path,fn)
            extid = os.path.splitext(fn)[0]
            pattern = FilePattern(self.env.newid(),self.source,extid,fp)
            if not self.framecontroller.exists(self.source,extid):
                try:
                    LOGGER.info(DiskAcquirer.processing_message(self.source, fn, i, lf))
                    addr = \
                        self.framecontroller.append(self.extractor_chain(pattern))
                    frames_processed += len(addr)
                except:
                    # KLUDGE: Do some real logging here
                    # TODO: How do I recover from an error once partial data has
                    # been written?
                    LOGGER.exception(traceback.format_exc())
            else:
                LOGGER.info(DiskAcquirer.skip_message(self.source, fn))
        
        return self.env.frames_to_seconds(frames_processed)
            
            
            
    
    