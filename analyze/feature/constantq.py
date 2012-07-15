from __future__ import division
from scikits.audiolab import Sndfile
import numpy as np
from scipy.signal.filter_design import butter
from scipy.signal import lfilter
from scipy.sparse import csr_matrix
from nputil import pad
from visualize.plot import plot

# TODO: This is a special case of the toeplitz2d function.  Create a generalized,
# cythonized version of this.
def buf(signal,windowsize,overlap):
    step = int(windowsize - overlap)
    n = []
    for i in xrange(0,len(signal),step):
        w = pad(signal[i:i + windowsize],windowsize)
        n.append(w)
    return  np.array(n)

def nextpow2(x):
    return int(np.ceil(np.log(float(x))/np.log(2.0)))

def sqrt_blackman_harris(size):
    '''
    square root of the blackman-harris window, defined at
    http://en.wikipedia.org/wiki/Window_function#Blackman.E2.80.93Harris_window
    '''
    r = np.arange(size)
    s = (size - 1)
    a0 = .35875
    a1 = .48829 * np.cos((2*np.pi*r)/s)
    a2 = .14128 * np.cos((4*np.pi*r)/s)
    a3 = .01168 * np.cos((6*np.pi*r)/s)
    return np.sqrt(a0 - a1 + a2 - a3)
    

class Kernel(object):
    
    def __init__(self,samplerate,max_freq,bins_per_octave):
        self._samplerate = samplerate
        self._max_freq = max_freq
        
        self._bins_per_octave = bins_per_octave
        # The threshold below which magnitudes in the kernel will be zeroed
        self._thresh = 5e-3
        # TODO: What does this do?
        self._quality = 1
        # TODO: What does this do?
        self._atom_hop_factor = .25
        self.q = 1
        # TODO: Explain each of these steps
        fmin = (self._max_freq/2) * (2**(1/self._bins_per_octave))
        self.Q = (1/(2**(1/self._bins_per_octave)-1)) * self._quality
        qsr = self.Q * self._samplerate
        self.largest_atom = np.round((qsr) / fmin)
        self.smallest_atom = np.round(qsr /\
                                 (fmin*(2**((self._bins_per_octave-1)/self._bins_per_octave))))
        self.atom_hop = np.round(self.smallest_atom*self._atom_hop_factor)
        hla = self.largest_atom/2
        first_center = np.ceil(hla)
        self.first_center = self.atom_hop * np.ceil(first_center/self.atom_hop)
        self.fftlen = 2**nextpow2(self.first_center + np.ceil(hla))
        self.atoms_per_frame = \
        np.floor((self.fftlen - np.ceil(hla)-self.first_center)/self.atom_hop) + 1
        last_center = self.first_center + ((self.atoms_per_frame-1) * self.atom_hop)
        self.fft_hop = (last_center + self.atom_hop) - self.first_center
        temp_kernel = np.zeros(self.fftlen,dtype = np.complex128)
        spar_kernel = []
        for b in xrange(self._bins_per_octave):
            bratio = b/self._bins_per_octave
            fk = fmin*(2**bratio)
            atom_size = np.round(qsr / fk)
            win = sqrt_blackman_harris(atom_size)
            temp_kernel_bin = (win/atom_size) * \
                np.exp((2*np.pi*1j*fk*np.arange(atom_size))/self._samplerate)
            atom_offset = self.first_center - np.ceil(atom_size/2)
            for a in xrange(int(self.atoms_per_frame)):
                shift = atom_offset + (a*self.atom_hop)
                temp_kernel[shift: shift + atom_size] = temp_kernel_bin
                spec_kernel = np.fft.fft(temp_kernel)
                spec_kernel[np.abs(spec_kernel) <= self._thresh] = 0
                spar_kernel.append(spec_kernel)
                temp_kernel[:] = 0
        
        
        spar_kernel = np.array(spar_kernel)
        spar_kernel = spar_kernel.T / self.fftlen
        minarg =  np.argmax(spar_kernel[:,0])
        maxarg =  np.argmax(spar_kernel[:,-1])
        wk = spar_kernel[minarg : maxarg,:]
        wk = np.diag(np.dot(wk,wk.T))
        z = np.round(1/self.q)
        wk = wk[z : -z-2]
        weight = 1/np.mean(np.abs(wk))
        weight = np.sqrt(weight*(self.fft_hop/self.fftlen))
        spar_kernel*= weight
        # TODO: This should be a sparse matrix
        self.kernel = np.array(np.matrix(spar_kernel).H)
    
    @property
    def overlap(self):
        return self.fftlen - self.fft_hop

class ConstantQ(object):
    
    def __init__(self,samplerate = 44100.,bins_per_octave = 24):
        object.__init__(self)
        self._samplerate = samplerate
        self._bins_per_octave = bins_per_octave
        self._max_freq = samplerate/3
        self._min_freq = self._max_freq/512
        
        # The total number of octaves the transform will cover
        self._noctaves = np.ceil(np.log2(self._max_freq/self._min_freq))
        # TODO: Why does this have to be recalculated
        self._min_freq = (self._max_freq/(2**self._noctaves)) *\
                         (2**(1/self._bins_per_octave))
        self._lowpass = butter(6,.5,'low')
        
        self._kernel = None
    
    def decimate(self,signal):
        '''
        Lowpass filter, then drop half the samples
        '''
        signal = lfilter(self._lowpass[0],self._lowpass[1],signal)
        return signal[::2]
        
    def transform(self,signal):
        if self._kernel is None:
            self._kernel = \
                Kernel(self._samplerate,self._max_freq,self._bins_per_octave)
        t = []
        max_block = self._kernel.fftlen * (2**self._noctaves)
        padded = \
            np.concatenate([np.zeros(max_block),signal,np.zeros(max_block)])
        overlap = self._kernel.overlap
        k = self._kernel.kernel
        
        for i in xrange(int(self._noctaves)):
            xx = buf(padded,self._kernel.fftlen,overlap)
            XX = np.fft.fft(xx)
            # KLUDGE: I have no idea if this is correct
            t.append(np.dot(k,XX.T))
            if i < self._noctaves:
                padded = self.decimate(padded)
        
        print '----------------------------------------'
        
        bpo = self._bins_per_octave
        no = self._noctaves
        to = bpo * no
        apf = self._kernel.atoms_per_frame
        empty_hops = self._kernel.first_center/self._kernel.atom_hop
        drop = (empty_hops*(2**(no-1))) - empty_hops
        cq = np.zeros((to,(t[0].shape[1] * apf) - drop))
        
        for i in xrange(int(no)):
            drop = (empty_hops * (2**(no - (i+1)))) - empty_hops
            sig = t[i]
            oct = np.zeros((bpo,(apf*sig.shape[1]) - drop))
            for b in xrange(bpo):
                start = b * apf
                stop = start + apf
                octbin = sig[start : stop]
                oct[b,:] = octbin.ravel()[drop:]
            sig = oct
            ipo = i + 1
            binstart = to - (bpo * ipo)
            binstop = to - (bpo * i)
            # BUG: yvec contains out-of-bounds indices for octaves > 0. Why?
            yvec = np.arange(0,(sig.shape[1] * (2**i)),(2**i)).astype(np.int32)
            cq[binstart : binstop,yvec] = sig
        
        cq = csr_matrix(cq)
        plot(cq,'cq')
            
                
    
    def __str__(self):
        return '''
samplerate      : %i
bins per octave : %i
min freq        : %1.4f
max freq        : %1.4f
octaves         : %1.4f
''' % (self._samplerate,
       self._bins_per_octave,
       self._min_freq,
       self._max_freq,
       self._noctaves)
        

if __name__ == '__main__':
    snd = Sndfile('/home/john/Test2/affected_male_speech.wav')
    sig = snd.read_frames(snd.nframes)
    cq = ConstantQ()
    cq.transform(sig)