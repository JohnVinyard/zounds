from __future__ import division
from uuid import uuid4
from multiprocessing import cpu_count

from zounds.analyze.synthesize import WindowedAudioSynthesizer
from zounds.util import tostring


class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1024
    window = None


class Environment(object):
    '''
    An Environment instance encapsulates all the settings required for a zounds
    application. It includes information about how audio will be processed and
    stored, which audio features will be computed, and which data backends
    will be used to store zounds.model.* instances.
    
    Instantiating an :py:class:`Environment` "wires up" your application, letting
    all the classes you'll be using know how audio should be processed, what
    features to extract, and how to store them.
    
    Normally, the :py:class:`Environment` will be instantiated in your config.py
    file.  Here's the example config.py file from the 
    :doc:`quickstart tutorial <quick-start>`, so you can see how the 
    :py:class:`Environment` class is used in context::
    
        # import zounds' logging configuration so it can be used in this application
        from zounds.log import *
        
        # User Config
        source = 'myapp'
        
        # Audio Config
        class AudioConfig:
            samplerate = 44100
            windowsize = 2048
            stepsize = 1024
            window = None
        
        # FrameModel
        from zounds.model.frame import Frames, Feature
        from zounds.analyze.feature.spectral import FFT,BarkBands
        
        class FrameModel(Frames):
            fft = Feature(FFT, store = False)
            bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
        
        
        # Data backends
        from zounds.model.framesearch import ExhaustiveSearch
        from zounds.model.pipeline import Pipeline
        from zounds.data.frame import PyTablesFrameController
        from zounds.data.search import PickledSearchController
        from zounds.data.pipeline import PickledPipelineController
        
        data = {
            ExhaustiveSearch    : PickledSearchController(),
            Pipeline            : PickledPipelineController()
        }
        
        
        from zounds.environment import Environment
        dbfile = 'datastore/frames.h5'
        Z = Environment(
                        source,                  # name of this application
                        FrameModel,              # our frame model
                        PyTablesFrameController, # FrameController class
                        (FrameModel,dbfile),     # FrameController args
                        data,                    # data-backend config
                        audio = AudioConfig)     # audio configuration
    
    Notice that the entire script consists of:
        
        * creating data structures which define different aspects of the application's behavior
        * passing those data structures to :py:meth:`Environment.__init__`
    '''
    
    n_cores = cpu_count()
    _test = False 
    instance = None
    
    @classmethod
    def parallel(cls):
        return not cls._test and cls.n_cores > 1
    
    def __new__(cls, *args, **kwargs):
        if not cls.instance or cls._test:       
            cls.instance = super(Environment, cls).__new__(cls)
        
        return cls.instance
        
    
    def __init__(self,source,framemodel,framecontroller,framecontroller_args,
                data,audio = AudioConfig,chunksize_seconds = 45.,do_sync = True):
        
        '''__init__
        
        :param source: The name of the application
        
        :param framemodel: A :py:class:`~zounds.model.frame.Frames`-derived class, which \
        defines the features to be computed
        
        :param framecontroller: A zounds.data.frame.FrameController-derived \
        class which handles the persistence, retrieval, and indexing of audio features.
        
        :param framecontroller_args: A tuple of arguments to be passed to the \
        framecontroller class' __init__ method
        
        :param data: A dictionary-like object mapping other zounds.model.* \
        classes to data backend instances
        
        :param audio: A class or object with attributes defining the sample \
        rate, window and step size, and the window (e.g., blackman-harris) to be applied to windows of audio samples prior to any processing.
        
        :param chunksize_seconds: The number of seconds of audio that will be \
        passed to the root extractor at a time
        
        :param do_sync: Mostly for internal use. This should usually be True.
        ''' 
        object.__init__(self)
        
        self.do_sync = do_sync
        
        #: audio settings, samplerate, windowsize and stepsize
        self.audio = audio
        
        self.chunksize_seconds = chunksize_seconds
        # processing chunk size, in samples
        self.chunksize = chunksize_seconds * self.samplerate
        # approximate number of absolute frames in each chunk
        self.chunksize_frames = int(self.chunksize / self.stepsize)
        
        # A synthesizer that can create raw audio samples from the encoding
        # stored in the 'audio' feature of frames.  Note that for now, the
        # "encoding" and "decoding" are really no-ops, except for the windowing
        # function on the encoding side, and the overlap/add on the decoding side
        self.synth = WindowedAudioSynthesizer(self.windowsize,self.stepsize)
        
        
        # the name of the 'source' attribute of patterns. Usually the client
        # application's name
        self.source = source
        
        #: a frame.model.Frames derived class that defines the features that
        #: the client app considers important
        self.framemodel = framemodel
        
        self.framecontroller_class = framecontroller
        self._framecontroller_args = framecontroller_args
        
        self._data = dict([(k,v.__class__) for k,v in data.iteritems()])
        # a dictionary-like object mapping classes to data-backends instances
        self.data = data
        
        self.framecontroller = framecontroller(*framecontroller_args)
        self.data[framemodel] = self.framecontroller
        if not Environment._test and self.do_sync:
            self.framemodel.sync()
    
    def __repr__(self):
        return tostring(self,
                        short = False,
                        sample_rate = self.samplerate,
                        window_size = self.windowsize,
                        step_size = self.stepsize,
                        chunksize_seconds = self.chunksize_seconds,
                        source = self.source,
                        framemodel = self.framemodel,
                        framecontroller = self.framecontroller_class,
                        data = self.data)
    
    def __str__(self):
        return tostring(self,
                        sample_rate = self.samplerate,
                        window_size = self.windowsize,
                        step_size = self.stepsize,
                        source = self.source,
                        framemodel = self.framemodel)
    
    def play(self,audio,block = True):
        '''play
        
        Play some audio.
        
        :param audio: the audio attribute of a \
        :py:class:`~zounds.model.frame.Frames`-derived instance
        :param block: If True, block until the audio is finished playing, \
        otherwise, play the audio in the background.
        '''
        self.synth.play(audio,block = block)
    
    def shush(self):
        '''shush
        
        Silence *all* currently playing audio 
        '''
        self.synth.shush()
    
    @property
    def address_class(self):
        return self.framecontroller_class.Address
        
    @property
    def windowsize(self):
        '''
        The window-size, in samples, to be used when processing audio
        '''
        return self.audio.windowsize
    
    @property
    def stepsize(self):
        '''
        The step-size, in samples, to be used when processing audio
        '''
        return self.audio.stepsize
    
    @property
    def samplerate(self):
        '''
        The sample rate, in hz, to be used when processing audio.  Incoming \
        sounds with a different sample rate will be re-sampled prior to processing.
        '''
        return self.audio.samplerate
    
    @property
    def window(self):
        '''
        The windowing function (e.g. blackman-harris) to be applied to windows \
        of audio samples prior to processing.
        '''
        return self.audio.window
    
    @property
    def overlap(self):
        '''
        The number of samples by which successive frames of audio overlap
        '''
        return self.windowsize - self.stepsize
    
    def seconds_to_frames(self,secs):
        '''seconds_to_frames
        
        Convert seconds to frames, i.e., windows of audio, using the current \
        audio settings
        
        :param secs: The number of seconds to convert to frames
        :returns: An integer number of frames
        '''
        return int((secs * self.samplerate) / self.stepsize)
    
    def frames_to_seconds(self,nframes):
        '''frames_to_seconds
        
        Convert a number of frames, i.e., windows of audio, to seconds using the\
        current audio settings
        
        :param nframes: The number of frames to convert to seconds
        :returns: The number of frames corresponding to seconds, given the \
        current audio settings.
        '''
        if not nframes:
            return 0
        
        return self.frames_to_samples(nframes) / self.samplerate
    
    def frames_to_samples(self,nframes):
        '''frames_to_samples
        
        Convert a number of frames, i.e., windows of audio, to samples using the\
        current audio settings
        
        :param nframes: The number of frames to convert to audio samples
        :returns: The number of samples corresponding to nframes, given the \
        current audio settings
        '''
        return (nframes * self.stepsize) + self.overlap
    
    def newid(self):
        return uuid4().hex
    
    def extractor_chain(self,pattern):
        return self.framemodel.extractor_chain(pattern)
    
    def append(self,pattern):
        ec = self.extractor_chain(pattern)
        self.framecontroller.append(ec)
    
    def start_audio_engine(self):
        self.synth._start_audio_engine()
    
    # KLUDGE: This is specific to PyTables, i.e., there are lots of problems
    # with concurrent reads.  This should be handled some other way.
    def unique_controller(self):
        return self.framecontroller_class(*self._framecontroller_args)
    
    def __getstate__(self,lock = None,**kwargs):
        d = {'source' : self.source,
             'framemodel' : self.framemodel,
             'framecontroller' : self.framecontroller_class,
             'framecontroller_args' : self._framecontroller_args,
             'data' : self._data,
             'audio' : self.audio,
             'chunksize_seconds' : self.chunksize_seconds,
             'do_sync' : self.do_sync}
        
        # KLUDGE: This assumes that lock will always be the final argument
        # to the FrameController
        if None is not lock:
            d['framecontroller_args'] += (lock,)
        
        for k,v in kwargs.iteritems():
            d[k] = v
        
        return d
    
    def __setstate__(self,d):
        data = d['data']
        for k,v in data.iteritems():
            # KLUDGE: This assumes that controllers will always have zero-arg
            # constructors
            #
            # BUG: For some reason, when n_cores = 2, the values in d['data']
            # come back as controller instances instead of classes when the
            # second round of pattern chunks are processed
            data[k] = v() if isinstance(v,type) else v

        return Environment(d['source'],
                           d['framemodel'],
                           d['framecontroller'],
                           d['framecontroller_args'],
                           data,
                           d['audio'],
                           d['chunksize_seconds'],
                           d['do_sync'])
    
    @staticmethod
    def reanimate(data):
        env = Environment.__new__(Environment)
        env = env.__setstate__(data)
        return env
    
    