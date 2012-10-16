from __future__ import division
import numpy as np
from scipy.signal import triang


def fft_index(freq_hz,ws,sr,rnd = np.round):
    '''
    Given a frequency in hz, a window size, and a sample rate,
    return the fft bin into which the freq in hz falls
    '''
    if freq_hz < 0 or freq_hz > sr / 2.: 
        raise ValueError(\
            'Freq must be greater than zero and less than the Nyquist frequency')
    
    fft_bandwidth = (sr * .5) / (ws * .5)
    return int(rnd(freq_hz / fft_bandwidth))

def fft_span(start_hz,stop_hz,ws,sr):
    '''
    Given a span in hz, return the start and stop fft bin indices covering
    that span
    '''
    s_index = fft_index(start_hz,ws,sr, rnd = np.floor)
    e_index = fft_index(stop_hz,ws,sr, rnd = np.ceil)
    # ensure that the span is at least one fft bin
    e_index = s_index + 1 if s_index == e_index else e_index
    return s_index,e_index

def _hz_is_valid(freq_hz,sr):
    if freq_hz < 0 or freq_hz > sr / 2.: 
        raise ValueError(\
            'Freq must be greater than zero and less than the Nyquist frequency')
    


def hz_to_barks(hz):
    return 6. * np.log((hz/600.) + np.sqrt((hz/600.)**2 + 1))

def barks_to_hz(b):
    return 300. * ((np.e ** (b/6.0)) - (np.e ** (-b/6.)))

def erb(hz):
    '''
    equivalent rectangular bandwidth
    '''
    return (0.108 * hz) + 24.7



def critical_bands(samplerate,
                   window_size,
                   fft_frame,
                   n_bark_bands,
                   start_freq = 50, 
                   stop_freq = 20000):

    # convert the start and stop freqs into the bark scale
    sb = hz_to_barks(start_freq)
    eb = hz_to_barks(stop_freq)
    # get the bandwidth (in barks), of each
    bark_bandwidth = (eb - sb) / n_bark_bands
    cb = np.ndarray(n_bark_bands)
    for i in xrange(1,n_bark_bands + 1):
        b = i * bark_bandwidth
        hz = barks_to_hz(b)
        _erb = erb(hz)
        start_hz = hz - (_erb/2.)
        start_hz = 0 if start_hz < 0 else start_hz
        s_index = fft_index(start_hz,window_size,samplerate)
        e_index = fft_index(hz + (_erb/2.),window_size,samplerate) + 1
        cb[i - 1] = (abs(fft_frame[s_index : e_index]) * triang(e_index - s_index)).sum()
        
    return cb


def tri_window(start_hz,stop_hz,ws,sr):
    # Idea 1: Scale by the max
    # Idea 2: Scale by the max and the diff
    
    # check that the hz values are greater than zero and less than the nyquist
    # frequency for this sampling rate
    _hz_is_valid(start_hz,sr)
    _hz_is_valid(stop_hz,sr)
    # How many fft bins equal 1hz?
    fft_bandwidth = (sr * .5) / (ws * .5)
    # the float-valued fft start bin
    s = start_hz / fft_bandwidth
    # the float-valued fft stop bin
    e = stop_hz / fft_bandwidth
    # the int-valued, rounded fft start bin
    s_index = int(np.floor(s))
    # the int-valued, rounded fft stop bin
    e_index = int(np.ceil(e))
    # make sure we're spanning at least one bin
    e_index = s_index + 1 if s_index == e_index else e_index
    # how many whole fft bins will this bark band use?
    nbins = e_index - s_index
    
    # a triangular window which we'll multiply the fft bins by, and then sum
    tri = triang(nbins)
    # ratio between 1 (the expected peak value for the triangular window) and
    # the actual value.  For windows with an even number of bins, this will 
    # always be > 1
    peak_diff = (1 / tri.max())
    # the difference between the real-valued fft bin span, and the rounded, 
    # int-valued fft bin span
    bin_diff = (e - s) / nbins
    tri = tri * peak_diff * bin_diff
    
    
    print peak_diff, bin_diff, tri.sum()
    
    # how much distortion did we introduce by rounding to a whole number of bins?
    #scale = (e - s) / nbins
    # scale the entire window by that much
    #tri *= scale
    
    
    return s_index,e_index,tri 

def bark_data(samplerate,window_size,nbands,start_freq_hz,stop_freq_hz):
    start_bark = hz_to_barks(start_freq_hz)
    stop_bark = hz_to_barks(stop_freq_hz)
    bark_bandwidth = (stop_bark - start_bark) / nbands
    # slices of fft coefficients
    _slices = []
    # triangle windows to multiply the fft slices by
    _triwins = []
    for i in xrange(1,nbands + 1):
        b = i * bark_bandwidth
        hz = barks_to_hz(b)
        _herb = erb(hz) / 2.
        start_hz = hz - _herb
        start_hz = 0 if start_hz < 0 else start_hz
        stop_hz = hz + _herb
        
        #s_index,e_index = fft_span(\
        #                start_hz,stop_hz,window_size,samplerate)
        
        #triang_size = e_index - s_index
        #print triang_size   
        #triwin = triang(triang_size)
        s_index,e_index,triwin = tri_window(\
                                start_hz,stop_hz,window_size,samplerate)
        _slices.append(slice(s_index,e_index))
        _triwins.append(triwin)
    return _slices,_triwins

def bark_bands(\
        samplerate,window_size,nbands,start_freq_hz,stop_freq_hz,fft,nframes,
        slices,triwins):
    
    cb = np.ndarray((nframes,nbands),dtype=np.float32)
    for i in xrange(nbands): 
        cb[:,i] = (fft[:,slices[i]] * triwins[i]).sum(1)
    return cb