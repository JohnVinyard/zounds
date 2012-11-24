import numpy as np
#from scikits.audiolab import play
from zounds.pattern import start,usecs,put,stop,cancel_all
from threading import Thread
from time import sleep


# KLUDGE: All these transforms should be written in C/C++, so they can be used with
# the realtime JACK player as well

# BUG: the todict() fromdict() cycle isn't idempotent!!
class Transform(object):
    
    JUMP = 0
    LINEAR = 1
    EXPONENTIAL = 2
    
    '''
    Transform some audio
    
    A transform applies to a single event.  The transform may take zero or more
    parameters at one or more instances in time.
    
    If a particular parameter has n (time,value) pairs defined, then there should
    be n-1 interpolations defined which determine how to travel between the two
    values.
    
    E.g, for an event that is one second long, I might apply a Gain transform.  
    Gain has a single parameter, "gain", which ranges from 0 to 1.  I might
    define the transform like this:
    
    Gain([(time = 0,value = 0),(time = 1, value = 1)],[Linear])
    
    In this case, the gain value will changed linearly from 0 to 1 over the 
    course of the sample.
    
    When multiple parameters are present, they can be changed independently.
    
    So, if
        - 0 values are present for a parameter, a sensible default is used
        - 1 value is present, this value is used for the duration of the event
        - 2 or more values are present, n -1 interpolations must be provided
    '''
    
    _impl = {}
    
    def __init__(self):
        object.__init__(self)
        self._impl[self.__class__.__name__] = self.__class__
    
    @property
    def args(self):
        return ()
    
    # TODO: Get rid of this!!  Clients should do the interpretation here
    def _normalize(self,param):
        '''
        Parameters may be passed as a single value, or an iterable of tuples of
        (values, times, interpolation types).  Normalize them.
        '''
        if hasattr(param,'__iter__'):
            values,times,interps = param
            return [values,(0,) + times,(Transform.LINEAR,) + interps]
        
        return[(param,),(0,),(Transform.JUMP)]
    
    def todict(self):
        return (self.__class__.__name__ ,self.args)
    
    @classmethod
    def fromdict(cls,d):
        return cls._impl[d[0]](*d[1])
    
    def _transform(self,audio):
        raise NotImplemented()
    
    def __call__(self,audio):
        return self._transform(audio)

class NoOp(Transform):
    
    def __init__(self):
        Transform.__init__(self)
    
    def _transform(self,audio):
        return audio
        

#TODO: This should accept constant and variable rate amplitude data too, defined
# by a zounds.timeseries.TimeSeries-derived class
class Gain(Transform):
    '''
    Adjust the amplitude of audio
    '''
    def __init__(self,gain):
        '''__init__
        
        :param amp: Either a single floating point value between 0 and 1, or an \
        iterable of tuples of ((values,),(times,),(interpolation types),)
        '''
        Transform.__init__(self)
        self.gain = self._normalize(gain)
    
    @property
    def args(self):
        return (self.amp,)
    
    def _transform(self,audio):
        raise NotImplemented()

class Delay(Transform):
    
    def __init__(self,dtime,feedback,level):
        Transform.__init__(self)
        self.dtime = self._normalize(dtime)
        self.feedback = self._normalize(feedback)
        self.level = self._normalize(level)
    
    @property
    def args(self):
        return (self.dtime,self.feedback,self.level)
    
class Convolver(Transform):
    ROOM = 0
    PLATE = 1
    HALL = 2
    
    def __init__(self,rtype,normalize,mix):
        Transform.__init__(self)
        self.rtype = rtype
        self.normalize = normalize
        self.mix = self._normalize(mix)
    
    @property
    def args(self):
        return (self.rtype,self.normalize,self.mix)

class BiQuadFilter(Transform):
    LOWPASS = 0
    HIGHPASS = 1
    BANDPASS = 2
    LOWSHELF = 3
    HIGHSHELF = 4
    PEAKING = 5
    NOTCH = 6
    ALLPASS = 7
    
    def __init__(self,ftype,frequency,q,gain):
        Transform.__init__(self)
        self.ftype = ftype
        self.frequency = self._normalize(frequency)
        self.q = self._normalize(q)
        self.gain = self._normalize(gain)
    
    @property
    def args(self):
        return (self.ftype,self.frequency,self.q,self.gain)

# TODO: How does this work?
class WaveShaper(Transform):
    
    def __init__(self):
        Transform.__init__(self)
        raise NotImplemented()
    
    @property
    def args(self):
        raise NotImplemented()

class Compressor(Transform):
    
    def __init__(self,threshold,knee,ratio,reduction,attack,release):
        Transform.__init__(self)
        self.threshold = self._normalize(threshold)
        self.knee = self._normalize(knee)
        self.ratio = self._normalize(ratio)
        self.reduction = self._normalize(reduction)
        self.attack = self._normalize(attack)
        self.release = self._normalize(release)
    
    @property
    def args(self):
        return (self.threshold,self.knee,self.ratio,
                self.reduction,self.attack,self.release)

class TransformChain(Transform):
    
    def __init__(self,transformers):
        Transform.__init__(self)
        self._chain = transformers
    
    def _transform(self,audio):
        ac = audio.copy()
        print self._chain
        for c in self._chain:
            ac = c(ac)
        return ac

    def __iter__(self):
        return self._chain.__iter__()
    
    def todict(self):
        return [c.todict() for c in self]
    
    @classmethod
    def fromdict(cls,d):
        return TransformChain([Transform.fromdict(z) for z in d])




# TODO: It'd probably be better to do a max size in bytes, rather than an 
# expiration time in minutes
class Buffers(Thread):
    
    def __init__(self,expire_time_minutes = 5):
        Thread.__init__(self)
        self.setDaemon(True)
        self._should_run = True
        self._buffers = dict()
        self._expire_seconds = expire_time_minutes * 60
        self.start()
    
    @property
    def env(self):
        return Environment.instance
    
    def has_key(self,key):
        return self._buffers.has_key(key)
    
    def _update(self,key):
        _,audio = self._buffers[key]
        self._buffers[key] = (time(),audio)
        return audio
    
    def __getitem__(self,key):
        return self._update(key)
    
    def allocate(self,p):
        
        try:
            self._update(p._id)
            return
        except KeyError:
            pass
        
        t = time()
        
        try:
            # p is a FrameModel-derived instance
            audio = self.env.synth(p.audio)
            # BUG: What if this is only a partial segment of a pattern?
            self._buffers[p._id] = (t,audio)
            return
        except AttributeError as e:
            print e
            pass
        
        try:
            # p is a leaf pattern, or a pattern that has been analyzed
            fm = self.env.framemodel
            frames = fm[p.address]
            audio = self.env.synth(frames.audio)
            self._buffers[p._id] = (t,audio)
            return
        except AttributeError as e:
            print e
            pass
        
        raise ValueError('p must be a FrameModel-derived instance or a Zound')
    
    def run(self):
        while self._should_run:
            keys = self._buffers.keys()
            tm = time()
            for k in keys:
                t,_ = self._buffers[k]
                if tm - t > self._expire_seconds:
                    print 'expiring %s' % k
                    del self._buffers[k]
            
            # check for expired buffers once a minute
            sleep(1000 * 60)
                
    
    def stop(self):
        self._should_run = False

class BufferBabysitter(Thread):
    '''
    This class is for internal use only.  This class solves the problem, albeit
    in a bit of a klunky way, of passing numpy arrays (which are subject to 
    garbage collection) into my JACK event queue.  If no reference to the
    numpy array of audio samples exists in python code, it's likely that the
    array will be garbage collected before, or during the time the audio is
    playing. C code will then attempt to access the memory, and a segfault
    will occur. This class is meant to solve the problem of one-off previews
    of audio, e.g.
    
    >>> frames = FrameModel.random()
    >>> Environment.instance.play(frames.audio)
    
    Note that audio will be synthesized from frames.audio and passed along to 
    the jack client, but no reference will remain in python code.
    
    A *real* buffer management scheme will be required when we start scheduling
    audio as musical patterns.
    '''
    def __init__(self,buf):
        Thread.__init__(self)
        self._buf = buf
        self.daemon = True
        # BUG: This only works for 44.1Khz. How do I know what samplerate the
        # buffer is?
        self._seconds = len(buf) / 44100.
        self._pos = 0
        self._should_stop = False
        

    def run(self):
        while self._pos < self._seconds and not self._should_stop:
            sleep(.5)
            self._pos += .5
        
        self.cleanup()
    
    def cleanup(self):
        del self._buf
    
class WindowedAudioSynthesizer(object):
    '''
    A very simple synthesizer that assumes windowed, but otherwise raw frames
    of audio.
    '''
    def __init__(self,windowsize,stepsize):
        object.__init__(self)
        self.windowsize = windowsize
        self.stepsize = stepsize
        self._overlap = self.windowsize - self.stepsize
        
        # libvoribs 1.3.1 is causing a segmentation fault when writing samples
        # near this number (regardless of sample rate)
        self.vorbis_seg_fault_size = 45 * 44100
        # Writing samples to a ogg vorbis file in chunks eliminates the segfault,
        # and is faster in some cases (especially for longer files), but introduces
        # small, but noticeable clicks at the chunk boundaries.  Libvorbis 
        # seems to not be "crosslapping" at write boundaries. 
        self.vorbis_chunk_size = 25 * 44100
    
    def _start_audio_engine(self):
        start()
        self._start_audio_engine = self._start_audio_engine_done
        # KLUDGE: I don't know how to determine when the JACK server is ready
        sleep(1) 
    
    def _start_audio_engine_done(self):
        pass
    
    def _stop_audio_engine(self):
        stop()
    
    
    def _vorbis_write(self,frames,sndfile,output,transformer):
        cs = self.vorbis_chunk_size
        waypoint = 0
        for i,f in enumerate(frames):
            start = i * self.stepsize
            stop = start + self.windowsize
            output[start : stop] += transformer(f)
            if (stop - waypoint) > cs:
                # a sndfile was passed in, and we have enough data to write
                # a chunk of audio to the file
                sndfile.write_frames(output[waypoint : stop])
                # update the end of the last chunk written
                waypoint = stop
        
        if sndfile:
            # write any remaining data to disk
            sndfile.write_frames(output[waypoint:])
        
        return output
    
    def __call__(self,frames,sndfile = None,transformer = NoOp()):
        '''
        Parameters
            frames  - a two dimensional array containing frames of audio samples
            sndfile - Optional. A scikits.audiolab.Sndfile instance to which the samples
                      should be written.
        
        Returns
            output - a 1D array containing the rendered audio
        '''
        
        # TODO: logic to translate frames to samples is already defined in 
        # Environment.frames_to_samples.  It'd be better to use that method 
        # here, but that would introduce a circular depenency, since the
        # Environment module imports this one.  This code should be factored out
        # to a common location.
        nsamples = self._overlap + (len(frames) * self.stepsize)
        
        # allocate memory for the samples
        output = np.zeros(nsamples,np.float32)
        
        if sndfile and 'ogg' == sndfile.file_format:
            # The caller has requested that output audio be written to an ogg
            # vorbis file.  libvorbis 1.3.1 is currently causing segmentation faults
            # for larger files, so we'll need to handle this specially.
            return self._vorbis_write(frames, sndfile, output,transformer)
    
        
        for i,f in enumerate(frames):
            start = i * self.stepsize
            stop = start + self.windowsize
            output[start : stop] += transformer(f)

    
        if sndfile:
            sndfile.write_frames(output)
        
        return output
            
    def play(self,audio,block = True):
        output = self(audio)
        return self.playraw(output,block = block)
    
    def playraw(self,audio,block = True):
        if np.float32 != audio.dtype:
            audio = audio.astype(np.float32)
        
        # ensure that the audio engine is started. If it has already been started,
        # this call is a no-op
        self._start_audio_engine()
        # this will "babysit" a reference to the audio samples until they're
        # through playing
        bb = BufferBabysitter(audio)
        bb.start()
        # time is in microseconds
        put(audio,0,len(audio),usecs() + 1e4)
        
        if block:
            # block until the audio is done playing, unless a KeyboardInterrupt
            # is received first
            try:
                while bb.is_alive():
                    bb.join(5)
            except KeyboardInterrupt:
                bb._should_stop = True
                self.shush()
    
    def shush(self):
        cancel_all()
    
    def __del__(self):
        self._stop_audio_engine()
        

# TODO: FFT synthesizer
# TODO: DCT synthesizer
# TODO: ConstantQ synthesizer


