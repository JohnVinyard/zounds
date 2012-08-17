from __future__ import division
import web
import os
import shutil
from time import sleep
from random import choice

import numpy as np
from scikits.audiolab import oggwrite,Sndfile
from scipy.misc import imsave

from config import *
from zounds.data.frame import Address
from zounds.model.framesearch import ExhaustiveLshSearch

media_path = 'media'
images_path = os.path.join(media_path,'images')
audio_path = os.path.join(media_path,'audio')

# TODO: Do I need this?
#lock_path = 'lock.dat'

controller = Z.framecontroller
# TODO: This needs to be configurable from the command line when the server is
# started!  None of this (including the search class) should be hard coded!
search = ExhaustiveLshSearch(\
                'search/packed',FrameModel.packed,step = 30,nbits = 64)
_ids = controller.list_ids()

urls = (r'/audio/(?P<addr>\d+_\d+)','audio',
        r'/image/(?P<addr>\d+_\d+)','image')

def decode_address(addr):
    start,stop = [int(s) for s in addr.split('_')]
    return Address(slice(start,stop))

def encode_address(addr):
    return '%s_%s' % (addr.key.start,addr.key.stop)

# TODO: Do I need this?
#def acquire_lock():
#    while os.path.exists(lock_path):
#        sleep(0.01)
#    
#    f = open(lock_path,'w')
#    f.close()
#
#def release_lock():
#    os.remove(lock_path)

# TODO: Refactor common code out of audio and image classes
class audio(object):
    
    def __init__(self):
        object.__init__(self)
    
    def filename(self,addr):
        return os.path.join(audio_path,'%s.ogg' % addr)
    
    def make_file(self,fn,addr):
        addr = decode_address(addr)
        frames = FrameModel[addr]
        raw = Z.synth(frames.audio)
        oggwrite(raw,fn)
    
    def GET(self,addr = None):
        fn = self.filename(addr)
        if not os.path.exists(fn):
            self.make_file(fn,addr)
        
        oggfile = Sndfile(fn,'r')
        seconds = oggfile.nframes / oggfile.samplerate
        oggfile.close()
        web.header('Content-Type','audio/ogg')
        web.header('X-Content-Duration',str(seconds))
        web.header('Accept-Ranges','bytes')
        with open(fn,'rb') as f:
            data = f.read()
        web.header('Content-Length',len(data))
        return data

class image(object):
    
    def __init__(self):
        object.__init__(self)
    
    def filename(self,addr):
        return os.path.join(images_path,'%s.png' % addr)
    
    def make_file(self,fn,addr):
        addr = decode_address(addr)
        frames = FrameModel[addr]
        bark = frames.bark
        imsave(fn,np.rot90(bark))
    
    def GET(self,addr = None):
        fn = self.filename(addr)
        if not os.path.exists(fn):
            self.make_file(fn, addr)
        
        web.header('Content-Type','image/png')
        with open(fn,'rb') as f:
            data = f.read()
        web.header('Content-Length',len(data))
        return data


class zoundsapp(object):
    
    def __init__(self):
        object.__init__(self)
    
    def GET(self):
        qid = choice(_ids)
        addr = controller.address(qid)
        start,stop = addr.key.start,addr.key.stop
        l = stop - start
        # TODO: Query lengths should be specified in arguments when the webserver
        # is started
        minlength = 60
        maxlength = 60 * 4
        qlength = np.random.randint(minlength,maxlength)
        slack = l - qlength
        if slack > 0:
            qstart = np.random.randint(slack) + start
            qstop = qlength + qstart
            addr = Address(slice(qstart,qstop))
        # TODO: nresults should be specified by a command-line arg when the server
        # is started
        results = search.search(addr, nresults = 10)
        # TODO: Render the results to a template
            
            
            
            

app = web.application(urls, globals())

if __name__ == "__main__":
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    app.run()