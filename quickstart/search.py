from __future__ import division
import os.path
from random import choice
from time import time
import argparse
import sys

from config import *
from scikits.audiolab import Sndfile,play
from model.framesearch import *
from environment import Environment
from util import audio_files,pad
from analyze.audiostream import read_frames_mono



def parse_args():
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    # required arguments
    aa('--feature',help='the name of the feature to use')
    aa('--searchclass',help='the name of the FrameSearch-derived class to use')
    aa('--sounddir',help='directory from which queries will be drawn')
    # optional arguments
    aa('--minseconds',
       help='the minimum length of a query',
       type=float,
       default = 1.0)
    aa('--maxseconds',
       help='the maximum length of a query',
       type=float,
       default=5.0)
    aa('--rebuild',
       help='forces the search index to be rebuilt',
       default = False, 
       action = 'store_true')
    
    args = parser.parse_args()
    if args.maxseconds <= args.minseconds:
        raise ValueError('maxseconds must be greater than minseconds')
    return args

def get_query(path,files,minseconds,maxseconds):
    snd = Sndfile(os.path.join(path,choice(files)))
    lensecs = snd.nframes / snd.samplerate
    # TODO: 
    # If the sound is less than minseconds, return
    # the sound, padded with zeros so that it is 
    # minseconds long
    #
    # If the sound is greater than minseconds, but less
    # than maxseconds, choose a snippet between (minseconds,len(sound))
    #
    # If the sound is greater than maxseconds, return a snippet between
    # minseconds and maxseconds
    #
    # Ensure that the samples are mono, and are at the environment's
    # sampling rate
    

if __name__ == '__main__':
    
    args = parse_args()
    
    _id = 'search/%s' % args.feature    
    searchclass = eval(args.searchclass)
    
    if args.rebuild:
        del searchclass[_id]

    try:
        search = searchclass[_id]
    except KeyError:
        search = searchclass(_id,getattr(FrameModel,args.feature))
        search.build_index()
    
    
    # Pick a sound segment
    d = args.sounddir
    minframes = Z.seconds_to_frames(args.minseconds)
    maxframes = Z.seconds_to_frames(args.maxseconds)
    files = audio_files(d)
    
    while True:
        start = 0
        stop = 0
        while (stop - start) < (40 * Z.windowsize) or (stop - start) > (120 * Z.windowsize):
            files = os.listdir(d)
            shuffle(files)
            query_file = None

            for f in files:
                if f.endswith('wav'):
                    snd = Sndfile(os.path.join(d,f))
                    if 44100 == snd.samplerate:
                        query_file = snd
                        break

            samples = snd.read_frames(snd.nframes)
            if 2 == snd.channels:
                samples = samples.sum(1) / 2.
            start = np.random.randint(len(samples))
            stop = start + np.random.randint(6,len(samples) - start)
        print 'Sound file : %s' % f


        # Do the search and time it
        starttime = time()
        results = search.search(samples[start : stop])
        print 'search took %1.4f seconds' % (time() - starttime)
        
        
        for _id,addr in results:
            try:
                print _id,addr
                Environment.instance.play(FrameModel[addr].audio)
            except KeyboardInterrupt:
                continue