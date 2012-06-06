from config import *
from scikits.audiolab import Sndfile,play
import os.path
from random import shuffle
from model.framesearch import *
from time import time
from environment import Environment
import argparse
import sys

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa('--feature',help='the name of the feature to use')
    aa('--searchclass',help='the name of the FrameSearch-derived class to use')
    aa('--rebuild',
       help='forces the search index to be rebuilt',
       default = False, 
       action = 'store_true')
    
    args = parser.parse_args()
    
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
    d = '/home/john/snd/Test'
    
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