from __future__ import division
import numpy as np

from analyze.extractor import SingleInput
from environment import Environment
from nputil import safe_unit_norm,safe_log
from util import downsample
from scipy.fftpack import dct

# Debugging
#np.seterr(all = 'raise')

class Scatter(SingleInput):
    
        
    
    def __init__(self,needs = None, key = None):
        # KLUDGE: This is set explicitly to yield a signal whose length is a 
        # power of two, using the common windowsize = 2048, step = 1024 setting.
        #
        # The algorithm I'm adapting requires that the input signal have a power
        # of size 2, but I should look for ways to relax this constraint.
        self.nframes = 63
        self.sizein = (63 * 1024) + 1024 # 2**16, or about 1.5 seconds at 44100hz
        SingleInput.__init__(self,needs = needs, nframes = self.nframes, 
                             step = 30, key = key)
        
        # FILTERBANK OPTIONS
        # The number of integer cycles processed at each frequency bin
        self.Q = 16
        self.dilation_factor = 2**(1/self.Q)
        #self.log_spaced_filters = \
        #    np.floor(np.log(self.sizein/2/self.Q) / np.log(self.dilation_factor))
        self.log_spaced_filters = 80
        self.linear_spaced_filters = \
            np.floor(1/np.log(self.dilation_factor)-1)
        self.total_filters = self.log_spaced_filters + self.linear_spaced_filters
        self.cauchy_order = 2
        self.lowpass_width = .63 # Why?!?!
        
        # SCATTERING OPTIONS
        self.antialiasing = 1
        # AKA aa_psi
        self.path = 2
        self.delta = -self.Q
        self.order = 2
        
        
        self._filter_bank = None
        self._lowpass = None
    
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return (1232,128)
    
    def fix(self,a):
        '''
        Set nan and inf values to zero
        '''
        a[np.isnan(a)] = 0
        a[np.isinf(a)] = 0
        return a
    
    def omega(self,size,scale):
        return np.arange(float(size)) / size * (2 * scale)
    
    def cauchy_wavelet(self,size,order,scale):
        # KLUDGE: What does this represent?
        omega = self.omega(size, scale)
        f1 = omega**order
        f1 = self.fix(f1)
        f2 = self.fix(np.exp(-order*(omega-1)))
        w = f1*f2
        return w
    
    def cauchy_scaling(self,size,order,scale):
        # perform partial integration?!?!
        omega = self.omega(size, scale)
        f = np.zeros(len(omega))
        for i in range(order):
            f1 = self.fix(omega**i)
            f2 = self.fix((1/order) * np.exp(-order*(omega-1)))
            f = ((i/order) * f) + (f1*f2)
            
        # normalize and mirrorize
        f /= f[0]
        f[1:] += f[1:][::-1]
        return f
        
    def setup_filterbank(self):
        # Mystery value
        mv = 177.5 * ((self.Q/8)**2)
        # normalization parameter ?!?!?!
        normal = 1
        nresolutions = np.log2(self.sizein)
        # This will contain a filterbank for each resolution
        bank = np.zeros((nresolutions,self.total_filters),dtype = object)
        lowpass = []
        for i in xrange(int(nresolutions)):
            size = self.sizein / (2**i)
            # This will contain the filterbank for the current resolution
            offset = np.round(i/np.log2(self.dilation_factor))
            # Add the log-spaced filters
            for q in xrange(int(self.log_spaced_filters - offset)):
                f = normal * self.cauchy_wavelet(size, mv, self.dilation_factor**q)
                assert offset + q <= self.log_spaced_filters
                bank[i][offset + q] = f
            
            # KLUDGE: What does this number mean? It's repeated above
            z = np.sqrt(mv / self.cauchy_order)
            # Add any linear-spaced filters
            for j in xrange(int(self.linear_spaced_filters)):
                zj = z/(z-(j-1)*(z-1)/(self.linear_spaced_filters - 1))
                psik = normal * self.cauchy_wavelet(
                                    size, mv/(zj**2), 
                                    (self.dilation_factor**self.log_spaced_filters)\
                                    *zj*size/self.sizein)
                mx = np.max(psik)
                mxidx = np.argmax(psik)
                if mxidx <= size/2 and mx > .9:
                    bank[i][self.log_spaced_filters + j] = psik
            
  
            if 0 == i:
                n = np.zeros(size)
                for j in xrange(int(offset),int(self.log_spaced_filters)):
                    n += np.abs(bank[i][j]**2)
                normal = np.sqrt(2/np.max(n))
                
                # renormalize the previously computed features
                for q in xrange(int(offset),int(self.total_filters)):
                    bank[i][q] *= normal
                    
            lp = self.cauchy_scaling(\
                        size,
                        self.cauchy_order,
                        self.dilation_factor**self.log_spaced_filters*size/self.sizein/self.lowpass_width)
            lowpass.append(lp)
            
        
        self._filter_bank = bank
        self._lowpass = lowpass
                
    # KLUDGE: WTF is j?        
    def next_bands(self,j):  
        bwm = self.dilation_factor**(1/np.log2(self.dilation_factor)- self.delta)      
        #d = np.log(2*(self.Q/self.delta)) / np.log(self.dilation_factor)
        d = np.log(2*self.Q/bwm)/np.log(self.dilation_factor)
        if j < self.log_spaced_filters-d:
            nb = j + d
        elif j < self.log_spaced_filters:
            nb = self.log_spaced_filters + \
                  (self.linear_spaced_filters * \
                   (1 - (self.dilation_factor**(self.log_spaced_filters-j) * \
                    self.delta)/(2*self.Q)))
        else:
            nb = self.log_spaced_filters + \
                (self.linear_spaced_filters*(1-(self.delta/(2 * self.Q))))
        
        return np.ceil(nb-1e-6)
    
    def audio_downsampling(self,j,bwm):
        band = min(j,self.log_spaced_filters)
        log2a = np.log2(self.dilation_factor)
        log2Q = np.log2(2*self.Q)
        log2bwm = np.log2(bwm)
        return np.floor(band * log2a + log2Q - log2bwm + 1e-6)
    
    def downsample_fac(self,res,j):
        ds = self.audio_downsampling(j, 2**self.antialiasing) - res
        return max(0,ds)
    
    def downsample_fac_psi(self,res,j):
        ds = self.audio_downsampling(j, 2**self.path) - res
        return max(0,ds)
    
    
    class Signal(object):
        
        def __init__(self,signal,scale,resolution):
            object.__init__(self)
            self.signal = signal
            self.scale = scale
            self.resolution = resolution
    
    def fft(self,sig):
        return np.fft.fft(sig)
    
    def decompose(self,sig):
        nj = self.total_filters
        fourier = self.fft(sig.signal)
        res = sig.resolution
        prevj = -100 if sig.scale < 0 else np.mod(sig.scale,nj)
        children = []
        nb = self.next_bands(prevj)
        for j in xrange(int(max(0,nb)),int(self.total_filters)):
            ds = self.downsample_fac_psi(res,j)
            out = np.abs(self.conv(sig,fourier,self._filter_bank[res][j],2**ds))
            newscale = (sig.scale>=0)*sig.scale*nj + j
            children.append(Scatter.Signal(out,newscale,res + ds))
        
        return children
        
        
    
    def smooth(self,sig):
        sigf = self.fft(sig.signal)
        nj = self.total_filters
        ds = self.downsample_fac(sig.resolution, nj)
        return self.conv(sig,sigf,self._lowpass[int(sig.resolution)],2**ds)
    
    def conv(self,sig,sigf,flter,dsfac):
        return np.fft.ifft(sigf * flter)[::dsfac] * np.sqrt(dsfac)
    
    

    def _process(self):
        audio = Environment.instance.synth(self.in_data)
        audio = safe_unit_norm(audio)
        if None is self._filter_bank:
            self.setup_filterbank()
        # a list whose first dimension represents the order, and whose second
        # dimension represents the signals to be decomposed
        S = [[Scatter.Signal(audio,-1,0)]]
        out = [[]]
        orders = range(1, self.order + 2)
        for o in orders:
            S.append([])
            out.append([])
            for q,sig in enumerate(S[o - 1]):
                if o <= self.order:
                    children = self.decompose(sig)
                    S[o].extend(children)
                out[o - 1].append(self.smooth(sig))
        
        #order_one = safe_log(np.abs(np.array(out[1])))
        order_two = safe_log(np.abs(np.array(out[2]))) 
        out = order_two
        return out
        
        