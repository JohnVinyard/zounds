from __future__ import division
import numpy as np

from analyze.extractor import SingleInput
from environment import Environment
from nputil import safe_unit_norm

# Debugging
#np.seterr(all = 'raise')

class Scatter(SingleInput):
    
    class FilterBank(object):
        
        def __init__(self,resolution):
            object.__init__(self)
            self.resolution = resolution
            self.log_spaced = []
            self.linear_spaced = []
            self.lowpass = None
        
        def add_log_spaced(self,f):
            self.log_spaced.append(f)
        
        def add_linear_spaced(self,f):
            self.linear_spaced.append(f)
        
        @property
        def n_log(self):
            return len(self.log_spaced)
        
        @property
        def n_linear(self):
            return len(self.linear_spaced)      
        
        @property
        def n_filters(self):
            return self.n_log + self.n_linear  
        
        
    
    def __init__(self,needs = None, key = None):
        # KLUDGE: This is set explicitly to yield a signal whose length is a 
        # power of two, using the common windowsize = 2048, step = 1024 setting.
        #
        # The algorithm I'm adapting requires that the input signal have a power
        # of size 2, but I should look for ways to relax this constraint.
        self.nframes = 63
        self.sizein = (63 * 1024) + 1024 # 2**16, or about 1.5 seconds at 44100hz
        SingleInput.__init__(self,needs = needs, nframes = 1, step = 1, key = key)
        
        # FILTERBANK OPTIONS
        # The number of integer cycles processed at each frequency bin
        self.Q = 16
        self.dilation_factor = 2**(1/self.Q)
        self.log_spaced_filters = \
            np.floor(np.log(self.sizein/2/self.Q) / np.log(self.dilation_factor))
        self.linear_spaced_filters = \
            np.floor(1/np.log(self.dilation_factor)-1)
        self.cauchy_order = 2
        self.lowpass_width = .63 # Why?!?!
        
        # SCATTERING OPTIONS
        self.antialiasing = 1
        self.path = 2
        self.delta = -self.Q
        self.order = 2
        
        
        self._filter_bank = None
    
    
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return 1
    
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
            #f2 = self.fix(1/(order*np.exp(-order*(omega-1))))
            #f = (i/(order*f)) + (f1 * f2)
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
        # This will contain a filterbank for each resolution
        filters = []
        for i in xrange(np.log2(self.sizein)):
            size = self.sizein / (2**i)
            print size
            # This will contain the filterbank for the current resolution
            fb = Scatter.FilterBank(i)
            offset = np.round(i/np.log2(self.dilation_factor))
            print offset
            # Add the log-spaced filters
            for q in range(self.log_spaced_filters - offset):
                fb.add_log_spaced(normal * \
                        self.cauchy_wavelet(size,
                                            mv,
                                            self.dilation_factor**q))
            
            # KLUDGE: What does this number mean? It's repeated above
            z = np.sqrt(mv / self.cauchy_order)
            # Add any linear-spaced filters
            for j in range(self.linear_spaced_filters):
                zj = z/(z-(j-1)*(z-1)/(self.linear_spaced_filters - 1))
                psik = normal * self.cauchy_wavelet(
                                    size, mv/(zj**2), 
                                    (self.dilation_factor**self.log_spaced_filters)\
                                    *zj*size/self.sizein)
                mx = np.max(psik)
                mxidx = np.argmax(psik)
                if mxidx <= size/2 and mx > .9:
                    fb.add_linear_spaced(psik)
            
            # compute the normalization
            if 0 == i:
                n = np.zeros(size)
                for j in range(offset,fb.n_log):
                    n += np.abs(fb.log_spaced[j])**2
                normal = np.sqrt(2/np.max(n))
                
                # renormalize the previously computed filters
                for i in range(fb.n_log):
                    fb.log_spaced[i] *= normal
                for i in range(fb.n_linear):
                    fb.linear_spaced[i] *= normal
            
            fb.lowpass = self.cauchy_scaling(\
                            size,
                            self.cauchy_order,
                            self.dilation_factor**self.log_spaced_filters*size/self.sizein/self.lowpass_width)
            filters.append(fb)
            print '-----------------------------------------------------------'
        
        self._filter_bank = filters    
                
    # KLUDGE: WTF is j?        
    def next_bands(self,j):        
        d = np.log(2*(self.Q/self.delta)) / np.log(self.dilation_factor)
        if j < self.log_spaced_filters-d:
            nb = j + d
        elif j < self.log_spaced_filters:
            nb = self.log_spaced_filters + \
                  (self.linear_spaced_filters * \
                   (1 - (self.dilation_factor**(self.log_spaced_filters-j) * \
                    self.delta)/(2*self.Q)))
        else:
            nb = self.log_spaced_filters + (self.linear_spaced_filters*(1-(self.delta/(2 * self.Q))))
        
        return np.ceil(nb-1e-6)
    
    def downsample(self,j,res):
        band = min(j,self.log_spaced_filters)
        log2a = np.log2(self.dilation_factor)
        log2Q = np.log2(2*self.Q)
        log2bwm = 2**self.antialiasing
        
        ds = np.floor(band * log2a + log2Q - log2bwm + 1e-6)
        return max(0,ds - res)
    
    def fft(self,sig):
        return np.fft.fft(sig)
    
    def decompose(self,sig):
        fourier = self.fft(sig.signal)
        
    
    def smooth(self,sig):
        pass
    
    def conv(self):
        pass
    
    class Signal(object):
        
        def __init__(self,signal,scale,orientation,resolution):
            object.__init__(self)
            self.signal = signal
            self.scale = scale
            self.orientation = orientation
            self.resolution = resolution

    def _process(self):
        audio = Environment.instance.synth(self.in_data)
        audio = safe_unit_norm(audio)
        if not self._filter_bank:
            self.setup_filter_bank()
        # a list whose first dimension represents the order, and whose second
        # dimension represents the signals to be decomposed
        S = [[Scatter.Signal(audio,-1,0,0)]]
        out = [[]]
        orders = range(1, self.order + 1)
        for o in orders:
            S.append([])
            out.append([])
            for q,sig in enumerate(S[o - 1]):
                children = self.decompose(sig)
                S[o].extend(children)
                out[o - 1].append(self.smooth(sig))
        return out
        