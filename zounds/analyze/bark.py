from __future__ import division
import numpy as np
from scipy.signal import triang


def fft_index(freq_hz,ws,sr):
    '''
    Given a frequency in hz, a window size, and a sample rate,
    return the fft bin into which the freq in hz falls
    '''
    if freq_hz < 0 or freq_hz > sr / 2.: 
        raise ValueError(\
            'Freq must be greater than zero and less than the Nyquist frequency')
    
    fft_bandwidth = (sr * .5) / (ws * .5)
    return int(round(freq_hz / fft_bandwidth))

def fft_span(start_hz,stop_hz,ws,sr):
    '''
    Given a span in hz, return the start and stop fft bin indices covering
    that span
    '''
    s_index = fft_index(start_hz,ws,sr)
    e_index = fft_index(stop_hz,ws,sr)
    # ensure that the span is at least one fft bin
    e_index = s_index + 1 if s_index == e_index else e_index
    return s_index,e_index

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
            s_index,e_index = fft_span(\
                            start_hz,stop_hz,window_size,samplerate)
            triang_size = e_index - s_index
            triwin = triang(triang_size)
            _slices.append(slice(s_index,e_index))
            _triwins.append(triwin)
        return _slices,_triwins

def bark_bands(\
        samplerate,window_size,nbands,start_freq_hz,stop_freq_hz,fft,nframes,
        slices,triwins):
    
    if nframes == 1:
        fft = fft.reshape((1,fft.shape[0]))
    
    cb = np.ndarray((nframes,nbands),dtype=np.float32)
    for i in xrange(nbands):
        cb[:,i] = (fft[:,slices[i]] * triwins[i]).sum(1)
    return cb