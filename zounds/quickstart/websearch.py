from __future__ import division
import web
import os
import shutil
from time import sleep,time
from random import choice

import numpy as np
from scikits.audiolab import oggwrite,Sndfile
from scipy.misc import imsave,imresize
from matplotlib.cm import hot

from config import *
from zounds.data.frame import Address
from zounds.model.framesearch import ExhaustiveLshSearch

import collections

KEY, PREV, NEXT = range(3)

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[PREV]
            curr[NEXT] = end[PREV] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[NEXT] = next
            next[PREV] = prev

    def __iter__(self):
        end = self.end
        curr = end[NEXT]
        while curr is not end:
            yield curr[KEY]
            curr = curr[NEXT]

    def __reversed__(self):
        end = self.end
        curr = end[PREV]
        while curr is not end:
            yield curr[KEY]
            curr = curr[PREV]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = next(reversed(self)) if last else next(iter(self))
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

    def __del__(self):
        self.clear()                    # remove circular references



media_path = 'media'
images_path = os.path.join(media_path,'images')
audio_path = os.path.join(media_path,'audio')

# TODO: Do I need this?
#lock_path = 'lock.dat'

controller = Z.framecontroller
# TODO: This needs to be configurable from the command line when the server is
# started!  None of this (including the search class) should be hard coded!
search = ExhaustiveLshSearch['search/packed']
_ids = list(controller.list_ids())

nframes = len(controller)
minutes = Z.frames_to_seconds(nframes)
hours = (minutes // 60) / 60
minutes = minutes % 60
human_friendly_db_length = '%i hours and %i minutes' % (hours,minutes)

urls = (r'/audio/(?P<addr>\d+_\d+)','audio',
        r'/image/(?P<addr>\d+_\d+)/(?P<feature>[\w]+)','image',
        r'/zoundsapp','zoundsapp')

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

# TODO: I should be using the address class itself here!
class Result(object):
    
    def __init__(self,_id,start,stop,score):
        self.id = _id
        self.start = start
        self.stop = stop
        self.score = score
        self.blocks = 1
        self.scores = [self.score]
    
    @property
    def nframes(self):
        return self.stop - self.start
    
        
    def __hash__(self):
        return hash((self.id,self.start,self.stop))
    
    def __eq__(self,other):
        return self.id == other.id and\
             self.start == other.start and\
              self.stop == other.stop
    
    def __lt__(self,other):
        return self.start < other.start
    
    def __le__(self,other):
        return self.start <= other.start
    
    def __gt__(self,other):
        return self.start > other.start
    
    def __ge__(self,other):
        return self.start >= other.start
    
    @staticmethod
    def congeal(results):
        srt = sorted(results)
        
        #return Result(results[0]._id,
        #              results[0].start,
        #              results[-1].stop,
        #              sum([r.score for r in results]))
        
        r = Result(srt[0].id,srt[0].start,srt[0].stop,srt[0].score)
        out = [r]
        for i in range(1,len(srt)):
            if srt[i].start - out[-1].stop > 500:
                nr = Result(srt[i].id,srt[i].start,srt[i].stop,srt[i].score)
                out.append(nr)
            else:
                out[-1].stop = srt[i].stop
                out[-1].scores.append(srt[i].score)
                out[-1].blocks += 1
        return out
        

class Results(object):
    
    def __init__(self,query,results,tic):
        self.query = query
        
        #self.results = \
        #    [Result(_id,addr.key.start,addr.key.stop,0) for _id,addr in results]
        #self.results = list(OrderedSet(self.results))
        
        d = dict()
        score = 0
        for _id,addr in results:
            r = Result(_id,addr.key.start,addr.key.stop,score)
            try:
                d[_id].add(r)
            except KeyError:
                d[_id] = set([r])
            score += 1
        
        self.results = []
        for v in d.itervalues():
            self.results.extend(Result.congeal(list(v)))
        
        self.results = sorted(self.results,key = lambda a : min(a.scores))

        self.search_time = time() - tic
        self.brag = 'Searched %s of sound in %1.4f seconds' %\
             (human_friendly_db_length,self.search_time)
    
class audio(object):
    
    def __init__(self):
        object.__init__(self)
    
    def filename(self,addr):
        return os.path.join(audio_path,'%s.ogg' % addr)
    
    def make_file(self,fn,addr):
        addr = decode_address(addr)
        print addr
        frames = FrameModel[addr]
        raw = Z.synth(frames.audio)
        print 'starting oggwrite'
        oggwrite(raw,fn)
        print 'finished oggwrite'
    
    def GET(self,addr = None):
        print 'making audio'
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
        print 'done making audio'
        return data

class image(object):
    
    def __init__(self):
        object.__init__(self)
    
    def filename(self,addr,feature):
        return os.path.join(images_path,'%s_%s.png' % (addr,feature))
    
    def make_file(self,fn,addr,feature):
        addr = decode_address(addr)
        frames = FrameModel[addr]
        f = getattr(frames,feature)
        f *= 1./ f.max()
        f = imresize(f,(f.shape[0] * 2, f.shape[1] * 2))
        color = hot(f)
        imsave(fn,np.rot90(color))
    
    def GET(self,addr = None,feature = None):
        print 'making image'
        print addr
        fn = self.filename(addr,feature)
        if not os.path.exists(fn):
            self.make_file(fn, addr,feature)
        
        web.header('Content-Type','image/png')
        with open(fn,'rb') as f:
            data = f.read()
        web.header('Content-Length',len(data))
        print 'done making image'
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
        print addr
        tic = time()
        results = search.search(addr, nresults = 75)
        r = Results(addr,results,tic)
        # TODO: Render the results to a template
        render = web.template.render('static/templates')
        return render.zoundsapp(r)
            
            
            

app = web.application(urls, globals())

if __name__ == "__main__":
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    app.run()