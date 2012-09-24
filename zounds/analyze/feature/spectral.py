from __future__ import division

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis
from scipy.signal import triang,lfilter
from scipy.signal.filter_design import butter
from scipy.fftpack import dct

from zounds.environment import Environment
from zounds.analyze.extractor import SingleInput
import zounds.analyze.bark as bark
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
        
        


class FFT(SingleInput):
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
        
        if None is not self._inshape:
            a = np.array(self._inshape,dtype = np.float32)
            a[0] = int(a[0] / 2)
            self._dim = int(np.product(a))
        else:
            ws = Environment.instance.windowsize
            self._dim = int(ws / 2)
            self._inshape = (ws,)
        
        # create a list of slices to remove the zero-frequency term along
        # whichever axis we're computing the FFT
        self._slice = [slice(None) for i in xrange(len(self._inshape) + 1)]
        # this slice will remove the zero-frequency term
        self._slice[self._axis] = slice(1,None)

    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._dim
    
    def _process(self):
        data = self.in_data
        data = data.reshape((data.shape[0],) + self._inshape)
        # return the magnitudes of a real-valued fft along the axis specified,
        # excluding the zero-frequency term
        out = np.abs(np.fft.rfft(data,axis = self._axis)[self._slice])
        return flatten2d(out)
    
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
        
        self.nbands = nbands
        self.env = Environment.instance
        self.samplerate = self.env.samplerate
        self.windowsize = self.env.windowsize
        
        self.start_freq_hz = start_freq_hz
        self.stop_freq_hz = stop_freq_hz
        self.start_bark = bark.hz_to_barks(self.start_freq_hz)
        self.stop_bark = bark.hz_to_barks(self.stop_freq_hz)
        self.bark_bandwidth = (self.stop_bark - self.start_bark) / self.nbands
        self._build_data()

    def dim(self,env):
        return self.nbands
    
    @property
    def dtype(self):
        return np.float32
    
    def _build_data(self):
        # slices of fft coefficients
        self._slices = []
        # triangle windows to multiply the fft slices by
        self._triwins = []
        for i in xrange(1,self.nbands + 1):
            b = i * self.bark_bandwidth
            hz = bark.barks_to_hz(b)
            _herb = bark.erb(hz) / 2.
            start_hz = hz - _herb
            start_hz = 0 if start_hz < 0 else start_hz
            stop_hz = hz + _herb
            s_index,e_index = bark.fft_span(\
                            start_hz,stop_hz,self.windowsize,self.samplerate)
            triang_size = e_index - s_index
            triwin = triang(triang_size)
            self._slices.append(slice(s_index,e_index))
            self._triwins.append(triwin)
    
    def _process(self):
        fft = self.in_data
        return bark.bark_bands(self.samplerate, 
                               self.windowsize, 
                               self.nbands, 
                               self.start_freq_hz, 
                               self.stop_freq_hz, 
                               fft, 
                               fft.shape[0],
                               self._slices,
                               self._triwins)

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

class SpectralCentroid(SingleInput):
    '''
    .. epigraph::
        Indicates where the "center of mass" of the spectrum is. Perceptually, 
        it has a robust connection with the impression of "brightness" of a 
        sound.  It is calculated as the weighted mean of the frequencies 
        present in the signal, determined using a Fourier transform, with 
        their magnitudes as the weights...
    
        -- http://en.wikipedia.org/wiki/Spectral_centroid
    
    This feature's :code:`needs` parameter usually points to a feature which \
    computes spectral coefficients, such as :py:class:`FFT`, or \
    :py:class:`BarkBands`, e.g::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            bark = Feature(BarkBands, needs = FFT)
            centroid = Feature(SpectralCentroid, needs = bark)
    '''
    
    def __init__(self,needs = None,key = None):
        '''__init__
        
        :param needs: A feature which probably consists of spectral coefficients
        '''
        SingleInput.__init__(self,needs = needs,key = key)
        self._bins = None
        self._bins_sum = None
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data
        if self._bins is None:
            self._bins = np.arange(1,spectrum.shape[-1] + 1)
            self._bins_sum = np.sum(self._bins)
        
        return (spectrum*self._bins).sum(axis = 1) / self._bins_sum


class SpectralFlatness(SingleInput):
    '''
    .. epigraph::
        Spectral flatness or tonality coefficient, also known as Wiener 
        entropy, is a measure used in digital signal processing to characterize an
        audio spectrum. Spectral flatness is typically measured in decibels, and 
        provides a way to quantify how tone-like a sound is, as opposed to being 
        noise-like. The meaning of tonal in this context is in the sense of the 
        amount of peaks or resonant structure in a power spectrum, as opposed to 
        flat spectrum of a white noise. A high spectral flatness indicates that 
        the spectrum has a similar amount of power in all spectral bands - this 
        would sound similar to white noise, and the graph of the spectrum would 
        appear relatively flat and smooth. A low spectral flatness indicates that
        the spectral power is concentrated in a relatively small number of 
        bands - this would typically sound like a mixture of sine waves, and the
        spectrum would appear "spiky"...
        
        -- http://en.wikipedia.org/wiki/Spectral_flatness
    
    This feature's :code:`needs` parameter usually points to a feature which \
    computes spectral coefficients, such as :py:class:`FFT`, or \
    :py:class:`BarkBands`, e.g::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            bark = Feature(BarkBands, needs = FFT)
            centroid = Feature(SpectralFlatness, needs = bark)
    '''
    
    def __init__(self, needs = None, key = None):
        '''__init__
        
        :param needs: A feature which probably consists of spectral coefficients
        '''
        SingleInput.__init__(self,needs = needs, key = key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = self.in_data
        m = spectrum.mean(axis = 1)
        m[m == 0] = -1e5
        return (gmean(spectrum,axis = 1) / m)

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
        self._memory = np.zeros(self.size)
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.inshape
    
    def _process(self):
        indata = self.in_data
        output =  indata - self._memory
        self._memory = indata
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
        