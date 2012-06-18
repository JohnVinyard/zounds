from __future__ import division
import os.path
from random import choice
import argparse

from config import *
from scikits.audiolab import Sndfile,play
from model.framesearch import *
from environment import Environment
from util import audio_files
from nputil import pad
from analyze.audiostream import read_frames_mono
from analyze.resample import Resample
from time import time

class Zndfile(Sndfile):
    '''
    A handy wrapper around scikits.audiolab.Sndfile which allows clients to 
    address sections of the file in a samplerate-agnostic way. Additionally,
    it performs on-the-fly resampling to the Zounds environment's sample rate.
    '''
    def __init__(self,filepath,out_samplerate = Z.samplerate):
        '''
        out_samplerate - the sample rate that all samples read by this instance
                         should be returned in, if resampling is requested when
                         reading
        '''
        Sndfile.__init__(self,filepath)
        self._out_samplerate = out_samplerate
    
    @property
    def nseconds(self):
        '''
        The length of this audio file, in seconds
        '''
        return self.nframes / self.samplerate
    
    @property
    def need_resample(self):
        '''
        Return true if samples read should be resampled
        '''
        return self.samplerate != self._out_samplerate
    
    def read_frames_padded(self,total_length_secs,resample = True):
        '''
        Read frames from the file, ensuring that the segment returned is at
        least total_length_secs in length
        ''' 
        # total desired length in samples, at the file's sampling rate
        ts = total_length_secs * self.samplerate
        # total number of frames to read from the file
        nframes = self.nframes if total_length_secs > self.nseconds else ts
        # Read samples.  Sum to mono if necessary, and pad with zeros so the 
        # length of samples is ts
        samples = pad(read_frames_mono(self,nframes),ts)
        if resample and self.need_resample:
            # resampling was requested, and the output sample rate doesn't match
            # the file's sample rate.
            return \
            Resample(self.samplerate,self._out_samplerate).all_at_once(samples)
        
        # Resampling wasn't requested, or wasn't necessary. Return the samples
        # unmodified
        return samples
    
    def seek_seconds(self,seconds,whence = 0):
        '''
        Move n seconds from either 0 = beginning of file, 1 = current position,
        or 2 = end of file
        '''
        nsamples = seconds * self.samplerate
        self.seek(nsamples,whence) 

    
def parse_args():
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    # required arguments
    aa('--feature',
       help='the name of the feature to use',
       required = True)
    aa('--searchclass',
       help='the name of the FrameSearch-derived class to use',
       required = True)
    aa('--sounddir',
       help='directory from which queries will be drawn',
       required = True)
    # optional arguments
    aa('--minseconds',
       help='the minimum length of a query',
       type=float,
       default = 1.0)
    aa('--maxseconds',
       help='the maximum length of a query',
       type=float,
       default = 5.0)
    aa('--rebuild',
       help='forces the search index to be rebuilt',
       default = False, 
       action = 'store_true')
    
    args,leftover = parser.parse_known_args()
    # KLUDGE: I need to be able to pass arbitrary kwargs to the search class
    # via the command-line interface. There's probably a better way, but this
    # is my best shot for the moment.
    def convert_leftovers(l):
        d = {}
        for i in xrange(len(l)):
            if i % 2:
                # this is a value
                try:
                    # number,bool,list, etc.
                    l[i] = eval(l[i])
                except NameError:
                    # the value is a string. Leave it alone.
                    pass
                d[l[i - 1]] = l[i]
            else:
                # This is a key. Strip leading dashes
                l[i] = l[i].lstrip('-')
        return d
            
    searchclass_kwargs = convert_leftovers(leftover)
    if args.maxseconds <= args.minseconds:
        raise ValueError('maxseconds must be greater than minseconds')
    return args,searchclass_kwargs

def get_query(path,files,minseconds,maxseconds):
    # choose a random audio file from path
    snd = Zndfile(os.path.join(path,choice(files)))
    if snd.nseconds < minseconds:
        # The lenght of the chosen sound is less than minseconds.  Return the 
        #  whole sound, padded with zeros so it's minseconds long, and sampled
        # at the current environment's sampling rate.
        return snd.read_frames_padded(minseconds)
    
    if snd.nseconds > minseconds and snd.nseconds < maxseconds:
        maxseconds = snd.nseconds
    
    # choose a length in seconds for the query that falls between minseconds
    # and maxseconds
    lsecs = minseconds + (np.random.random() * (maxseconds - minseconds))
    # choose the starting point for the segment
    start = np.random.random() * (snd.nseconds - lsecs)
    # fetch the segment at the current environment's sampling rate
    snd.seek_seconds(start)
    return snd.read_frames_padded(lsecs)
    
    
if __name__ == '__main__':
    
    args,searchclass_kwargs = parse_args()
    
    _id = 'search/%s' % args.feature    
    searchclass = eval(args.searchclass)
    
    if args.rebuild:
        try:
            del searchclass[_id]
        except KeyError:
            # It's ok. No search with that name exists. We'll be rebuilding it
            # anyway.
            pass

    try:
        search = searchclass[_id]
    except KeyError:
        search = searchclass(\
                    _id,getattr(FrameModel,args.feature),**searchclass_kwargs)
        search.build_index()
    
    
    # Pick a sound segment
    d = args.sounddir
    files = audio_files(d)
    
    while True:
        # get a random query
        query = get_query(d,files,args.minseconds,args.maxseconds)
        # Do the search and time it
        starttime = time()
        results = search.search(query)
        print 'search took %1.4f seconds' % (time() - starttime)
        # play the query sound
        Z.synth.playraw(query)
        # play the search results
        for _id,addr in results:
            try:
                print _id,addr
                Environment.instance.play(FrameModel[addr].audio)
            except KeyboardInterrupt:
                continue