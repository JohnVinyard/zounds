#!/usr/bin/env python

from __future__ import division
import web
import os
from time import time
from random import choice

import numpy as np
from scikits.audiolab import Sndfile,Format
from scipy.misc import imsave,imresize
from matplotlib.cm import hot

from config import *
from zounds.util import ensure_path_exists,SearchSetup

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



controller = Z.framecontroller
args,search = SearchSetup(FrameModel).setup()

_ids = list(controller.list_ids())

nframes = len(controller)
minutes = Z.frames_to_seconds(nframes)
hours = (minutes // 60) / 60
minutes = minutes % 60
human_friendly_db_length = '%i hours and %i minutes' % (hours,minutes)

urls = (r'/audio/(?P<addr>.+?)','audio',
        r'/image/(?P<addr>.+?)/(?P<feature>[\w]+)','image',
        r'/zoundsapp','zoundsapp')


# KLUDGE: The way addresses are being handled is atrocious!
from zounds.data.frame.pytables import PyTablesFrameController
from zounds.data.frame.filesystem import FileSystemFrameController

Address = Z.address_class
if controller.__class__ == PyTablesFrameController:
    def decode_address(addr):
        start,stop = [int(s) for s in addr.split(':')]
        return Address(slice(start,stop))
    
    def encode_address(addr):
        return '%s:%s' % (addr.start,addr.stop)
    
    def build_address(_id,start,stop):
        return Address(slice(start,stop))

elif controller.__class__ == FileSystemFrameController:
    def decode_address(addr):
        _id,start,stop = addr.split(':')
        return Address((_id,slice(int(start),int(stop))))
    
    def encode_address(addr):
        return '%s:%s:%s' % (addr._id,addr.start,addr.stop)
    
    def build_address(_id,start,stop):
        return Address((_id,slice(start,stop)))
else:
    raise RuntimeError('Cannot handle %s backend' % controller.__class__.__name__)

class Tile(object):
    
    def __init__(self,_id,start,stop,start_offset = 0,stop_offset = 0):
        self.id = _id
        self.start = start
        self.stop = stop
        self.start_offset = start_offset
        self.stop_offset = stop_offset
        self.tilescale = 2
        
    @property
    def width(self):
        return ((self.stop - self.start) - self.start_offset - self.stop_offset)\
                 * self.tilescale
    
    def __repr__(self):
        return '''background-image : url('image/%s/bark'); background-position : -%ipx 0px; width : %ipx;''' % \
                 (encode_address(build_address(self.id,self.start,self.stop)),
                  self.start_offset * self.tilescale,
                  self.width)
    
    def __str__(self):
        return self.__repr__()

# TODO: I should be using the address class itself here!
class Result(object):
    
    def __init__(self,_id,start,stop,score):
        
        self.id = _id
        self.start = start
        self.stop = stop
        self.score = score
        self.blocks = 1
        self.scores = [self.score]
        
    
    def compute_tiles(self):
        # TODO: Should this be a part of the Tile class?
        self.tilesize = 30
        tilestart = int(np.floor(self.start / self.tilesize) * self.tilesize)
        tilestop = int(np.ceil(self.stop / self.tilesize) * self.tilesize)
        start_offset = self.start - tilestart
        stop_offset = tilestop - self.stop
        t = range(tilestart,tilestop,self.tilesize)
        self.tiles = []
        if 1 == len(t):
            self.tiles = [Tile(self.id,tilestart,tilestop,start_offset,stop_offset)]
        else:
            count = 0
            for i in t:
                stop = i + self.tilesize
                so = start_offset if count == 0 else 0
                eo = stop_offset if count == len(t) - 1 else 0
                self.tiles.append(Tile(self.id,i,stop,so,eo))
                count += 1
    
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
        [o.compute_tiles() for o in out]
        return out
    
    @property
    def encoded_address(self):
        return encode_address(build_address(self.id,self.start,self.stop))
        

class Results(object):
    
    def __init__(self,query_id,query,results,tic):
        self.query_id = query_id
        self.query = query
        self.querytile = Tile(self.query_id,query.start,query.stop)
        d = dict()
        score = 0
        for _id,addr in results:
            r = Result(_id,addr.start,addr.stop,score)
            try:
                d[_id].add(r)
            except KeyError:
                d[_id] = set([r])
            score += 1
        
        self.results = []
        for v in d.itervalues():
            self.results.extend(Result.congeal(list(v)))
        
        self.results = sorted(self.results,key = lambda a : np.mean(a.scores))

        self.search_time = time() - tic
        self.brag = 'Searched %s of sound in %1.4f seconds' %\
             (human_friendly_db_length,self.search_time)
    
    @property
    def query_address(self):
        return encode_address(build_address(self.query_id,self.query.start,self.query.stop))

# TODO: Refactor common code out of audio and image classes
class audio(object):
    
    def __init__(self):
        object.__init__(self)
    
    def filename(self,addr):
        return os.path.join(audio_path,'%s.ogg' % addr)
    
    def make_file(self,fn,addr):
        addr = decode_address(addr)
        frames = FrameModel[addr]
        fmt = Format('ogg','vorbis')
        sndfile = Sndfile(fn,'w',fmt,1,Z.samplerate)
        print 'starting oggwrite of %s' % addr
        Z.synth(frames.audio,sndfile)
        sndfile.close()
            
    
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
        fn = self.filename(addr,feature)
        if not os.path.exists(fn):
            self.make_file(fn, addr,feature)
        
        web.header('Content-Type','image/png')
        web.header('Cache-Control','private')
        
        web.http.expires(60 * 60 * 24 * 30)
        
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
        start,stop = addr.start,addr.stop
        l = len(addr)
        minlength = Z.seconds_to_frames(args.minseconds)
        maxlength = Z.seconds_to_frames(args.maxseconds)
        qlength = np.random.randint(minlength,maxlength)
        slack = l - qlength
        if slack > 0:
            qstart = np.random.randint(slack) + start
            qstop = qlength + qstart
            addr = build_address(qid,qstart,qstop)
        
        tic = time()
        results = search.search(addr, nresults = args.nresults)
        r = Results(qid,addr,results,tic)
        render = web.template.render('static/templates')
        return render.zoundsapp(r)
            
            
            
ensure_path_exists(images_path)
ensure_path_exists(audio_path)
app = web.application(urls, globals())
application = app.wsgifunc()    