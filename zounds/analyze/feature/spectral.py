from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis
from scipy.signal import lfilter
from scipy.signal.filter_design import butter
from scipy.fftpack import dct
from scipy.spatial.distance import cdist

from zounds.environment import Environment
from zounds.analyze.extractor import SingleInput
from zounds.analyze.psychoacoustics import Bark,Chroma as ChromaScale
from zounds.nputil import safe_log,flatten2d,sliding_window


class BandpassFilter(SingleInput):
    
    def __init__(self,needs = None, key = None, nframes = 1, step = 1,
                 start_band = None,stop_band = None,filter_order = 6):
        
        SingleInput.__init__(self, needs = needs, nframes = nframes, 
                             step = step, key = key)
        if start_band < 0 or start_band >= stop_band:
            raise ValueError('start band must be less than stop band')
        
        if stop_band > 1:
            raise ValueError('start_band and stop_band be between 0 and 1')
        
        self._start_band = start_band
        self._stop_band = stop_band
        print self._start_band,self._stop_band
        self._coeffs = butter(\
            filter_order,(self._start_band,self._stop_band),btype = 'bandpass')
        print np.abs(np.roots(self._coeffs[1]))
        assert np.all(np.abs(np.roots(self._coeffs[1]))<1)
        self._env = Environment.instance
        self._dim = self._env.windowsize
        self._window = self._env.window if None is not self._env.window else 1
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._dim
    
    def _process(self):
        # KLUDGE: This will produce discontinuities at chunk boundaries. I can
        # live with this for now. The right way is to look at how lfilter works,
        # and mantain whatever state is necessary to filter properly across
        # chunk boundaries
        data = self.in_data
        env = Environment.instance
        # data is windowed audio data. create an audio signal
        audio = env.synth(data)
        # bandpass filter the audio signal
        filtered = lfilter(self._coeffs[0],self._coeffs[1],audio)
        return sliding_window(filtered,env.windowsize,env.stepsize) * self._window
        
        
class SpectralDecomposition(SingleInput):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, needs = None, key = None, inshape = None, 
                 nframes = 1, step = 1, axis = -1, op = None, abs_val = True):

        SingleInput.__init__(self, needs = needs, nframes = nframes, 
                             step = step, key = key)
        self._axis = axis if axis < 0 else axis + 1
        self._dim = None
        self._slice = None
        if None is inshape or isinstance(inshape,tuple):
            self._inshape = inshape
        elif isinstance(inshape,int):
            self._inshape = (inshape,)
        else:
            raise ValueError('inshape must be None, an int, or a tuple')
        
        self._op = op
        self._compute_shape()
        self._compute_slice()
        self._finalize = np.abs if abs_val else lambda x : x
    
    @abstractmethod
    def _compute_shape(self):
        raise NotImplemented()
    
    @abstractmethod
    def _compute_slice(self):
        raise NotImplemented()
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._dim
    
    def _process(self):
        data = self.in_data
        data = data.reshape((data.shape[0],) + self._inshape)
        out = self._finalize(self._op(data,axis = self._axis)[self._slice])
        return flatten2d(out)

class FFT(SpectralDecomposition):
    '''
    Compute the real-valued magnitudes of input data.  This means that
        * phase is discarded
        * the zero-frequency term is discarded
        * all magnitudes are positive
    
    The most common use for this class will be to compute the STFT of audio \
    frames.  This is the default behavior::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
    
    It is possible to apply this class to other signals.  To apply an FFT to the
    STFT we just computed, similar to the frequency-invariance-promoting \
    operation performed when computing \
    `MFCCs or BFCCs <http://en.wikipedia.org/wiki/Mel-frequency_cepstrum>`_, we
    can do the following::
    
        class FrameModel(Frames):
            # compute the STFT
            fft = Feature(FFT)
            # take an FFT of the spectrum itself.  Note that this example 
            # assumes an audio configuration with a window size of 2048 samples.
            # The output shape of the "fft" feature will be 1024, and the output
            # shape of this feature will be 512.
            cepstrum = Feature(FFT, needs = fft, inshape = 1024)
    
    We could also try to find periodicities in the overall amplitude of the 
    signal, like so::
    
        class FrameModel(Frames):
            # compute the STFT
            fft = Feature(FFT)
            # Compute the "loudness" of the signal by summing each FFT frame
            loudness = Feature(Loudness, needs = fft)
            # Each time we've collected 60 frames of loudness data, apply a 
            # fourier transform.  This spectrum should display strong peaks if
            # the signal is periodic over time.
            periods = Feature(FFT, needs = loudness, inshape = 60, nframes = 60, step = 30)
    '''
    
    def __init__(self, needs = None, key = None, inshape = None, 
                 nframes = 1, step = 1, axis = -1):
        '''__init__
        
        :param needs: the single feature we'll be applying a fourier transform to
         
        :param nframes: the number of frames to collect before performing a computation
        
        :param step: step size. This determines the overlap between successive frames
        
        :param inshape: The shape of individual frames of incoming data. This must \
        be specified if this extractor is doing something other than its default \
        behavior, which is to compute an STFT of audio frames whose size, overlap \
        and windowing function is determined by the current \
        :py:class:`~zounds.environment.Environment` instance.
        
        :param axis: The axis of :code:`self.in_data` over which the FFT should \
        be computed.  The default is the last axis.
        '''
        SpectralDecomposition.__init__(self, needs = needs, key = key,
                                       inshape = inshape, nframes = nframes,
                                       step = step, axis = axis, 
                                       op = np.fft.rfft)
    
    def _compute_shape(self):
        if None is not self._inshape:
            a = np.array(self._inshape,dtype = np.float32)
            a[0] = int(a[0] / 2)
            self._dim = int(np.product(a))
        else:
            ws = Environment.instance.windowsize
            self._dim = int(ws / 2)
            self._inshape = (ws,)
    
    def _compute_slice(self):
        # create a list of slices to remove the zero-frequency term along
        # whichever axis we're computing the FFT
        self._slice = [slice(None) for i in xrange(len(self._inshape) + 1)]
        # this slice will remove the zero-frequency term
        self._slice[self._axis] = slice(1,None)


class DCT(SpectralDecomposition):
    
    def __init__(self, needs = None, key = None,inshape = None,
                 nframes = 1, step = 1, axis = -1, abs_val = True):
        
        def op(x, axis = axis):
            return dct(x, norm = 'ortho', axis = axis)

        SpectralDecomposition.__init__(self, needs = needs, key = key,
                                       inshape = inshape, nframes = nframes,
                                       step = step, axis = axis, 
                                       op = op, abs_val = abs_val)
        
    def _compute_shape(self):
        self._dim = self._inshape = \
            self._inshape if self._inshape else (Environment.instance.windowsize,)
        
    def _compute_slice(self):
        self._slice = slice(None)


class Chroma(SingleInput):
    
    def __init__(self, needs = None, key = None,
                 nbins = 12, a440 = 440.):
        SingleInput.__init__(self,needs=needs, nframes=1, step=1, key=key)
        self.env = Environment.instance
        self.nbins = nbins
        self.a440 = a440
        self.chroma = ChromaScale(\
                    self.env.samplerate,self.env.windowsize,nbands = 12)
    
    def dim(self,env):
        return self.nbins
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return self.chroma.transform(self.in_data)
    
class BarkBands(SingleInput):
    '''
    Maps STFT coefficients computed from raw audio data onto the \
    `Bark psychoacoustical scale <http://en.wikipedia.org/wiki/Bark_scale>`_.  \
    While most of the energy in the FFT is concentrated in the lower-frequency
    coefficients, energy will be more evenly distributed across the Bark \
    coefficients.
    
    Most of the time, :code:`BarkBands` will be used like this:::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            bark = Feature(BarkBands, needs = fft)
    '''
    
    def __init__(self,needs=None, key=None,
                 nbands = 100,start_freq_hz = 50, stop_freq_hz = 20000):
        '''__init__
        
        :param needs:  The source feature. This should usually be a \
        :py:class:`FFT` feature with the default behavior
        
        :param nbands: The resolution of the Bark spectrum
        
        :param start_freq_hz: FFT coefficients representing frequencies below \
        this value will be excluded.
        
        :param stop_freq_hz: FFT coefficients representing frequencies above \
        this value will be excluded.
        '''
        
        SingleInput.__init__(self,needs=needs, nframes=1, step=1, key=key)
        self.env = Environment.instance
        self.bark = Bark(self.env.samplerate,self.env.windowsize,nbands,
                         start_freq_hz,stop_freq_hz)

    def dim(self,env):
        return self.bark.n_bands
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return self.bark.transform(self.in_data)

class Loudness(SingleInput):
    '''
    Loudness performs no psychoacoustical weighting of input frequencies, and
    simply sums input data over one or more dimensions.  It should probably
    be replaced by a more general :code:`Sum` class.
    
    It can be used to find the magnitude of individual frames of FFT \
    coefficients, e.g.::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            loudness = Feature(Loudness, needs = fft)
    
    Or, it can be used to sum over both frequency and time, like so::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            loudness = Feature(Loudness, needs = fft, nframes = 10, step = 4)
    '''
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        '''__init__
        
        :param needs: The feature to be summed
        
        :param nframes: The number of frames from the source feature to sum over
        
        :param step: The number of frames of the input feature described by a \
        single frame of this feature
        '''
        SingleInput.__init__(self,needs=needs,nframes=nframes,step=step,key=key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
        
    def _process(self):
        ls = len(self.in_data.shape)
        r = range(ls)
        summed = np.apply_over_axes(np.sum, self.in_data, r[1:])
        if summed.size > 1:
            return summed.squeeze()
        
        return summed.reshape((1,))

class Kurtosis(SingleInput):
    
    def __init__(self, needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return kurtosis(self.in_data,axis = 1)
    
class BFCC(SingleInput):
    '''
    BFCC stands for Bark Frequency Cepstral Coefficients. It is very similar to \
    the popular \
    `MFCC, or Mel Frequency Cepstral Coefficients <http://en.wikipedia.org/wiki/Mel-frequency_cepstrum>`_, \
    only differing \
    in the psychoacoustical scale onto which FFT coefficients are mapped prior \
    to computing the cepstrum.  The aim of both features is to describe the shape \
    of the spectrum in a frequency-invariant way.  Concretely (and ideally), a \
    saxophone playing two different notes, perhaps in different octaves, would
    still have the same spectral "shape".  The feature is computed as follows:
        
        * Take an STFT of audio data
        * map the fourier coefficients onto the Bark scale
        * Take the log() of the coefficients, to decrease dynamic range
        * Take a DCT of the log of the coefficients.  This is essentially a \
          spectrum of the fourier spectrum.  Typically, only the first 12 or 13 \
          DCT coefficients are kept, since the majority of the signal's energy \
          is concentrated in the these lower frequencies.
    
    For the sake of decomposability and DRY-ness, the :py:class:`BFCC` class does \ 
    not implement the first two steps itself, but expects :py:class:`BarkBands` \
     as input::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            bark = Feature(BarkBands, needs = fft, nbands = 200)
            bfcc = Feature(BFCC, needs = bark)
    '''
    
    def __init__(self, needs = None, key = None, ncoeffs = 13,exclude = 1):
        '''__init__
        
        :param needs: A :py:class:`BarkBands` feature
        
        :param ncoeffs: The number of DCT coefficients to keep. 12 or 13 is a \
        commonly chosen number
        
        :param exclude: How many of the bottom DCT coefficients to discard. It's \
        typical to discard the first coefficient, since it can roughly be \
        interpreted as the signal's overall loudness.
        '''
        SingleInput.__init__(self, needs = needs, key = key)
        self.ncoeffs = ncoeffs
        # the first coefficient is often dropped, as it carries information
        # about the loudness of the signal
        self.exclude = exclude
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.ncoeffs
    
    def _process(self):
        barks = self.in_data
        
        return dct(safe_log(barks),axis = 1)[:,self.exclude: self.exclude + self.ncoeffs]

# TODO: Write docs for this demonstrating how to find periodicities in a 
# loudness input
class AutoCorrelation(SingleInput):
    '''
    Compute the autocorrelation, using the Wiener-Khinchin theorem, detailed
    here: http://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    '''
    def __init__(self, needs = None, key = None, inshape = None):
        SingleInput.__init__(self, needs = needs, key = key)
        if not inshape:
            raise ValueError('please specifiy a size')
        self._inshape = inshape
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._inshape
    
    def _process(self):
        data = self.in_data
        f = np.fft.fft(data,axis=-1)
        f2 = f*f.conjugate()
        return np.fft.ifft(f2,axis = -1)

# TODO: Consider adding an option to normalize (i.e. give unit-norm) to frames
# to de-emphasize volume changes and emphasize spectral changes alone, as suggested
# here: http://en.wikipedia.org/wiki/Spectral_flux
class Difference(SingleInput):
    '''
    Compute the first-order difference between the current frame of features and 
    a previous frame in any number of dimensions.  Typically, this will serve 
    as a precursor to the :py:class:`Flux` feature.  The first frame is also
    compared to zeros of the same dimension as the input feature.
    
    To calculate the change in loudness over time::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            loudness = Feature(Loudness, needs = fft)
            diff = Feature(Difference, needs = loud)
    
    To calculate the difference between successive FFT frames, a precursor to
    computing spectral flux::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            # this assumes a window size of 2048
            diff = Feature(Difference, needs = FFT, inshape = 1024)
    '''
    
    def __init__(self, needs = None, key = None, inshape = ()):
        '''__init__
        
        :param needs: The feature to compute the first-order difference over
        
        :param inshape: The shape of individual input frames. By default, a \
        one-dimensional signal is expected
        '''
        SingleInput.__init__(self, needs = needs, key = key)
        self.inshape = inshape
        self._memory = None
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.inshape
    
    def _process(self):
        if self._memory is None:
            self._memory = self.in_data[0]
        # prepend memory to data
        data = np.concatenate([self._memory[None,...],self.in_data])
        # take diff
        output = np.diff(data,axis = 0)
        # set memory to be last item in data
        self._memory = data[-1]
        # return output
        return output

class Flux(SingleInput):
    '''
    Compute the Euclidean norm of the difference between successive vectors.  For \
    the sake of decomposability, this extractor expects :py:class:`Difference` as \
    an input.  To compute spectral flux::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            # this assumes a window size of 2048
            diff = Feature(Difference, needs = fft, inshape = 1024)
            flux = Feature(Flux, needs = diff)
    '''
    
    def __init__(self, needs = None, key = None):
        '''__init__
        
        :param needs: The feature whose euclidean norm should be computed
        '''
        SingleInput.__init__(self, needs = needs, key = key)
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return ()
    
    def _process(self):
        return np.sqrt((self.in_data**2).sum(axis = 1))


class SelfSimilarityMatrix(SingleInput):
    '''
    Return the upper diagonal of a self similarity matrix for a feature, 
    excluding the middle/diagonal indices, which are, by definition, zero.
    
    Use euclidean distance by default
    '''
    def __init__(self,needs = None, key = None, inshape = None):
        SingleInput.__init__(self, needs = needs, key = key)
        self._indices = np.triu_indices(inshape,k = 1)
    
    @property
    def dtype(self):
        return np.float32

    def dim(self,env):
        return (self._indices[0].size,)
    
    def _process(self):
        o = np.zeros(\
            (self.in_data.shape[0],self._indices[0].size))
        data = self.in_data
        data = data if len(data) == 3 else data[:,:,None]
        # TODO: Can this be vectorized?
        for i,d in enumerate(data):
            o[i] = cdist(d,d)[self._indices]
        return o
        