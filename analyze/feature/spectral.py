from __future__ import division
from itertools import product

import numpy as np

from scipy.stats.mstats import gmean
from scipy.stats import kurtosis
from scipy.signal import triang
from scipy.fftpack import dct
from scipy.ndimage.interpolation import rotate
from scipy.spatial.distance import cdist
from scipy.signal import convolve

from environment import Environment
from analyze.extractor import SingleInput
from analyze import bark
from nputil import safe_log,safe_unit_norm as sun



class FFT(SingleInput):
    
    def __init__(self,needs=None,key=None):
        SingleInput.__init__(self,needs=needs,nframes=1,step=1,key=key)
    
    def dim(self,env):
        return int(env.windowsize / 2)
    
    @property
    def dtype(self):
        return np.float32
        
    def _process(self):
        '''
        Return the magnitudes only, discarding phase and the zero
        frequency component
        '''
        return np.abs(np.fft.rfft(self.in_data[0]))[1:]

# KlUDGE: Generalize this, and collapse into FFT
class FFT2(SingleInput):
    
    def __init__(self,inshape = None,needs=None,key=None,
                 nframes = 1,step = 1,axis = 0):
        
        SingleInput.__init__(self,needs = needs, nframes = nframes, 
                             step = step, key = key)
        
        if isinstance(inshape,int):
            self._inshape = (inshape,)
        elif isinstance(inshape,tuple):
            self._inshape = inshape
        else:
            raise ValueError('inshape must be an int or a tuple')
        
        self._axis = axis
    
    def dim(self,env):
        a = np.array(self._inshape,dtype = np.float32)
        a[0] = int(a[0] / 2)
        return int(np.product(a))
    
    @property
    def dtype(self):
        return np.float32

    def _process(self):
        data = np.array(self.in_data[:self.nframes]).reshape(self._inshape)
        # the slice here is to remove the zero-frequency term
        return [np.abs(np.fft.rfft(data,axis = self._axis)[1:]).ravel()]
    
class BarkBands(SingleInput):
    
    _triang = {}
    _fft_span = {}
    _hz_to_barks = {}
    _barks_to_hz = {}
    _erb = {}
    
    def __init__(self,needs=None, key=None,
                 nbands = None,start_freq_hz = 50, stop_freq_hz = 20000):
        
        SingleInput.__init__(self,needs=needs, nframes=1, step=1, key=key)
        if None is nbands:
            raise ValueError('an integer must be supplied for nbands')
        
        self.nbands = nbands
        self.env = Environment.instance
        self.samplerate = self.env.samplerate
        self.windowsize = self.env.windowsize
        
        self.start_freq_hz = start_freq_hz
        self.stop_freq_hz = stop_freq_hz
        self.start_bark = bark.hz_to_barks(self.start_freq_hz)
        self.stop_bark = bark.hz_to_barks(self.stop_freq_hz)
        self.bark_bandwidth = (self.stop_bark - self.start_bark) / self.nbands

    def dim(self,env):
        return self.nbands
    
    @property
    def dtype(self):
        return np.float32
    
    @classmethod
    def from_cache(cls,cache,call,key):
        try:
            return cache[key]
        except KeyError:
            v = call(key)
            cache[key] = v
            return v
        
    @classmethod
    def fft_span(cls,t):
        return bark.fft_span(*t)
    
    def _process(self):
        cb = np.ndarray(self.nbands,dtype=np.float32)
        for i in xrange(1,self.nbands + 1):
            b = i * self.bark_bandwidth
            hz = BarkBands.from_cache(BarkBands._barks_to_hz, 
                                      bark.barks_to_hz, 
                                      b)
            _erb = BarkBands.from_cache(BarkBands._erb, bark.erb, hz)
            _herb = _erb / 2.
            start_hz = hz - _herb
            start_hz = 0 if start_hz < 0 else start_hz
            stop_hz = hz + _herb
            ws = self.windowsize
            sr = self.samplerate
            s_index,e_index = BarkBands.from_cache(BarkBands._fft_span,
                                                   BarkBands.fft_span,
                                                   (start_hz,stop_hz,ws,sr))
            triang_size = e_index - s_index
            triwin = self.from_cache(BarkBands._triang,triang,triang_size)
            fft_frame = self.in_data[0]
            cb[i - 1] = \
                (fft_frame[s_index : e_index] * triwin).sum()
        
        return cb

class Loudness(SingleInput):
    
    def __init__(self,needs=None,nframes=1,step=1,key=None):
        SingleInput.__init__(self,needs=needs,nframes=nframes,step=step,key=key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
        
    def _process(self):
        return np.sum(self.in_data[:self.nframes])



class SpectralCentroid(SingleInput):
    '''
    "Indicates where the "center of mass" of the spectrum is. Perceptually, 
    it has a robust connection with the impression of "brightness" of a 
    sound.  It is calculated as the weighted mean of the frequencies 
    present in the signal, determined using a Fourier transform, with 
    their magnitudes as the weights..."
    
    From http://en.wikipedia.org/wiki/Spectral_centroid
    '''
    
    def __init__(self,needs = None,key = None,step = 1):
        SingleInput.__init__(self,needs = needs,key = key, step = step)
        self._bins = None
        self._bins_sum = None
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = np.array(self.in_data[0]).squeeze()
        if self._bins is None:
            self._bins = np.arange(1,len(spectrum) + 1)
            self._bins_sum = np.sum(self._bins)
            
        return np.sum(spectrum*self._bins) / self._bins_sum


class SpectralFlatness(SingleInput):
    '''
    "Spectral flatness or tonality coefficient, also known as Wiener 
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
    spectrum would appear "spiky"..."
    
    From http://en.wikipedia.org/wiki/Spectral_flatness
    '''
    
    def __init__(self, needs = None, key = None, step = 1):
        SingleInput.__init__(self,needs = needs, key = key, step = step)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        spectrum = np.array(self.in_data[0]).squeeze()
        avg = np.average(spectrum)
        # avoid divide-by-zero errors
        return gmean(spectrum) / avg if avg else 0

class Kurtosis(SingleInput):
    
    def __init__(self, needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
    
    def dim(self,env):
        return ()
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        return kurtosis(self.in_data[0])
    
class BFCC(SingleInput):
    
    def __init__(self, needs = None, key = None, ncoeffs = 13,exclude = 1):
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
        barks = self.in_data[0]
        return dct(safe_log(barks))[self.exclude: self.exclude + self.ncoeffs]

class AutoCorrelation(SingleInput):
    def __init__(self, needs = None, key = None, size = None):
        SingleInput.__init__(self, needs = needs, key = key)
        if not size:
            raise ValueError('please specifiy a size')
        self.size = size
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.size
    
    def _process(self):
        data = np.array(self.in_data[0]).reshape(self.size)
        return np.correlate(data,data,mode = 'full')[self.size - 1:]

class SelfSimilarity(SingleInput):
    def __init__(self,needs = None, key = None, dim = None):
        SingleInput.__init__(self,needs = needs, key = key)
        self._dim = dim
    
    @property
    def dtype(self):
        return np.float32

    def dim(self,env):
        return self._dim

    def _process(self):
        data = np.array(self.in_data[0])
        data = data.reshape((len(data),1))
        dist = np.rot90(cdist(data,data))
        dist += 1e-12
        return np.diag(dist)[self._dim:]
        
        
class Difference(SingleInput):
    def __init__(self, needs = None, key = None, size = None):
        SingleInput.__init__(self, needs = needs, key = key)
        if not size:
            raise ValueError('please specifiy a size')
        self.size = size
        self._memory = np.zeros(self.size)
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self.size
    
    def _process(self):
        indata = self.in_data[0]
        output =  indata - self._memory
        self._memory = indata
        return output

class Flux(SingleInput):
    
    def __init__(self, needs = None, key = None):
        SingleInput.__init__(self, needs = needs, key = key)
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return ()
    
    def _process(self):
        diff = self.in_data[0]
        return np.linalg.norm(diff)
    
    
class Intervals(SingleInput):
    
    def __init__(self,needs = None, key = None, nintervals = None):
        SingleInput.__init__(self,needs = needs, key = key)
        self.nintervals = nintervals
        # we're returning the top diagonals of the comparison matrix. The number
        # of elements is an arithmetic series.
        n = self.nintervals - 1
        self._dim = (n/2) * (n + 1)
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return int(self._dim)
    
    def _process(self):
        indata = self.in_data[0]
        # get the indices of the n most dominant coefficients
        top = np.argsort(indata)[-self.nintervals:]
        # get the intervals between each coefficient
        mat = np.array(np.abs(top - np.matrix(top).T))
        # return only the top diagonals of the matrix; everything else is
        # redundant
        return np.concatenate([np.diag(mat,i) for i in range(1,self.nintervals)])



# BUG: This class requires that the number of channels and number of frames be
# equal, i.e., the scale must be square.
class Gabor(SingleInput):
    
    def __init__(self,nchannels = None,needs = None, key = None):
        
        SingleInput.__init__(self,needs = needs, key = key, 
                             nframes = nchannels, step = 40)
        self.nchannels = nchannels
        self._indim = (nchannels,nchannels)
        # TODO: Decide these values based on the number of channels and 
        # number of frames
        self.freqs = [8,16,32,64]
        self.rotations = [0,30,60,90,120,150]
        self.scales = [10,15,20,25,30,25]
        self.x = np.arange(0,nchannels - 8,8)
        self.y = np.arange(0,nchannels - 8,8)
        self._lengths = [len(self.freqs),
                         len(self.rotations),
                         len(self.scales),
                         len(self.x),
                         len(self.y)] 
        self._dim = np.sum(self._lengths)
        self._filterbank = self.gabor_filterbank(self.freqs, 
                                                  self.rotations, 
                                                  self.scales, 
                                                  self.x, 
                                                  self.y)
        self._filterbank = self._filterbank.reshape(\
                        (np.product(self._lengths),nchannels * nchannels))
        
        
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return self._dim
    
    # KLUDGE: This method leaves rough, raw edges around the filter when it's
    # rotated
    def gabor(self,fullscale,scale,freq,rotation,location):
        # create a buffer to hold the gabor filter
        a = np.zeros((scale,scale))
        cps = freq / fullscale
        # create a 1d gabor filter at scale
        gabor1d = np.sin(np.arange(0,scale * cps,cps) * (2*np.pi))[:scale] * np.hamming(scale)
        # turn the 1d filter into a 2d filter
        a[:] = gabor1d
        a = a.T
        a *= np.hamming(scale)
        # rotate the gabor filter
        a = rotate(a,rotation,reshape = False)
        # place the gabor filter at the specified location on the fullsize filter
        b = np.zeros((fullscale,fullscale))
        x,y = location
        lx = len(b[x : x + scale])
        ly = b[:,y : y + scale].shape[1]
        b[x : x + scale , y : y + scale] = a[:lx,:ly]
        return b
    
    
    def gabor_filterbank(self,freqs,rotations,scales,xloc,yloc):
        l = np.product(self._lengths)
        d = np.zeros((l,self.nframes,self.nchannels))
        for i,frc in enumerate(product(freqs,rotations,scales,xloc,yloc)):
            freq,rot,scale,x,y = frc
            d[i] = self.gabor(self.nchannels,scale,freq,rot,(x,y))
            norm = np.linalg.norm(d[i])
            if norm:
                d[i] /= norm
        return d
    
    def _process(self):
        data = np.reshape(self.in_data,(self.nchannels ** 2))
        act = np.abs(np.dot(data,self._filterbank.T))
        act = act.reshape(self._lengths)
        sums = []
        for i in range(5):
            l = range(5)
            l.remove(i)
            s = np.apply_over_axes(np.sum, act, l)
            sums.append(s.ravel())
        # BUG: Why does this have to be returned as a list?
        return [np.concatenate(sums)]
    