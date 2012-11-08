import numpy as np
#from scikits.audiolab import play
from zounds.pattern import start,usecs,put,stop,cancel_all
from threading import Thread
from time import sleep


# KLUDGE: All these transforms should be written in C, so they can be used with
# the realtime JACK player as well
class Transform(object):
    '''
    Transform some audio
    '''
    
    _impl = {}
    
    def __init__(self):
        object.__init__(self)
        self._impl[self.__class__.__name__] = self.__class__
    
    @property
    def args(self):
        return ()
    
    def todict(self):
        return (self.__class__.__name__ ,self.args)
    
    @classmethod
    def fromdict(cls,d):
        return transformers[d[0]](*d[1])
    
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
class Amplitude(Transform):
    '''
    Adjust the amplitude of audio
    '''
    def __init__(self,amp):
        Transform.__init__(self)
        self.amp = amp
        
    @property
    def args(self):
        return (self.amp,)
    
    def _transform(self,audio):
        return audio * self.amp
    
# TODO: Dirac-based time and pitch stretch

# TODO: Basic low and high-pass filters

# TODO: Freeverb

class TransformChain(Transform):
    
    def __init__(self,transformers):
        Transform.__init__(self)
        self._chain = transformers
    
    def _transform(self,audio):
        ac = audio.copy()
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

# KLUDGE: There's gotta be a better way than this
transformers = {
    'Amplitude' : Amplitude
}

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


