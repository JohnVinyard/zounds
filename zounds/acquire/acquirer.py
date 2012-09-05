from __future__ import division
from abc import ABCMeta,abstractmethod,abstractproperty
import os.path
from time import time
from multiprocessing import Manager,Pool
import traceback

from zounds.constants import available_file_formats
from zounds.environment import Environment
from zounds.model.pattern import FilePattern


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
        return self.env.framemodel
    
    @property
    def framecontroller(self):
        return self.framemodel.controller()
    
    def extractor_chain(self,pattern):
        return self.framemodel.extractor_chain(pattern)
    
    @abstractproperty
    def source(self):
        '''
        The source name, e.g. 'Freesound' or 'StemsFromMyAlbum'
        '''
        pass
    
    @abstractmethod
    def _acquire(self):
        pass
    
    def acquire(self):
        '''
        Do the work necessary to acquire sound data, process it, and
        insert it into the db
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
                print DiskAcquirer.processing_message(\
                                            source, fn, filen + i, total_files)
                addr = Z.framecontroller.append(Z.framemodel.extractor_chain(pattern))
                total_frames += len(addr)
            except:
                # KLUDGE: Do some real logging here
                # TODO: How do I recover from an error once partial data has
                # been written? 
                print traceback.format_exc()
        else:
            print DiskAcquirer.skip_message(source, fn)
    return total_frames
    
class DiskAcquirer(Acquirer):
    
    def __init__(self,path,source = None):
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
        self.framecontroller.update_index()
        if not seconds_processed:
            print DiskAcquirer.no_files_processed_message(self._source)
        else:
            print DiskAcquirer.complete_message(seconds_processed, elapsed)    
    
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
                    print DiskAcquirer.processing_message(self.source, fn, i, lf)
                    addr = \
                        self.framecontroller.append(self.extractor_chain(pattern))
                    frames_processed += len(addr)
                except:
                    # KLUDGE: Do some real logging here
                    # TODO: How do I recover from an error once partial data has
                    # been written?
                    print traceback.format_exc()
            else:
                print DiskAcquirer.skip_message(self.source, fn)
        
        return self.env.frames_to_seconds(frames_processed)
            
            
            
    
    