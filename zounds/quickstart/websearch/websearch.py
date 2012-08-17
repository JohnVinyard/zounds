import web
import os
from time import sleep

from scikits.audiolab import oggwrite

from config import *
from zounds.model.data.frame import Address



media_path = 'media'
images_path = os.path.join(media_path,'images')
audio_path = os.path.join(media_path,'audio')
lock_path = 'lock.dat'

controller = Z.framecontroller 

urls = ('/(.*)', 'Hello',
        '/audio/(?P<addr>\d+_\d+)')

def decode_address(addr):
    start,stop = [int(s) for s in addr.split('_')]
    return Address(slice(start,stop))

def encode_address(addr):
    return '%s_%s' % (addr.key.start,addr.key.stop)

def acquire_lock():
    while os.path.exists(lock_path):
        sleep(0.01)
    
    f = open(lock_path,'w')
    f.close()

def release_lock():
    os.remove(lock_path)


class Hello:
    def GET(self, path=None):
        return 'Hello, world!'

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
        web.header('Content-Type','audio/ogg')
        web.header('Accept-Ranges','bytes')
        with open(fn,'rb') as f:
            data = f.read()
        return data

app = web.application(urls, globals())

if __name__ == "__main__":
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    app.run()