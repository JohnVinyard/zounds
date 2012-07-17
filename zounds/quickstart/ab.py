from __future__ import division
from config import *
import numpy as np
from zounds.nputil import pad
import cPickle
from random import choice
import sys


def load_data(filename):
    try:
        with open(filename,'r') as f:
            return cPickle.load(f)
    except IOError:
        return []

def save_data(data,filename):
    with open(filename,'w') as f:
        cPickle.dump(data, f)



def random_slice(_id,nseconds = 3):
    frames = FrameModel[_id]
    nsamples = nseconds * Z.samplerate
    if frames.seconds <= nseconds:
        samples = Z.synth(frames.audio)
        return _id,slice(len(frames)),pad(samples,nsamples)
    
    nframes = Z.seconds_to_frames(nseconds)
    
    first = len(frames) - nframes
    start = np.random.randint(0,first)
    stop = start + nframes
    samples = Z.synth(frames[start:].audio)
    padded = pad(samples,nsamples)[:nsamples]
    
    return _id,slice(start,stop),padded


def save_choice(data,q_id,qslice,a_id,aslice,b_id,bslice,aorb):
    data.append(((q_id,qslice),
                 (a_id,aslice),
                 (b_id,bslice),
                 aorb))
        
def main(data,nseconds = 3):
    print 'You have ranked %i' % len(data)
    _ids = list(FrameModel.list_ids())
    q_id = choice(_ids)
    raw_input('model')
    q_id,qslice,qpadded = random_slice(q_id) 
    Z.synth.playraw(qpadded)
    _ids.remove(q_id)
    
    a_id = choice(_ids)
    _ids.remove(a_id)
    raw_input('0')
    a_id,aslice,apadded = random_slice(a_id)
    Z.synth.playraw(apadded)
    
    b_id = choice(_ids)
    _ids.remove(b_id)
    raw_input('1')
    b_id,bslice,bpadded = random_slice(b_id)
    Z.synth.playraw(bpadded)
    
    selection = raw_input('0,1, or ?')
    
    try:
        aorb = int(selection)
        save_choice(data,q_id,qslice,a_id,aslice,b_id,bslice,aorb)
    except ValueError:
        pass
    
    print '--------------------------------------'
    


if __name__ == '__main__':
    filename = sys.argv[1]
    data = load_data(filename)
    for d in data:
        print d
    try:
        while True:
            main(data)
    except:
        save_data(data,filename)
        