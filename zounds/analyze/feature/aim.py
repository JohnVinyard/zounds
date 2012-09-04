import aimc
import numpy as np
from zounds.analyze.extractor import SingleInput
from zounds.environment import Environment
from zounds.util import flatten2d


class AIMFeature(SingleInput):
    '''
    Features that wrap up code from the aimc library
    '''
    
    def __init__(self,needs = None, key = None,highest_freq = 12000):
        SingleInput.__init__(self,needs = needs, nframes = 1, step = 1, key = key)
        env = Environment.instance
        if env.windowsize != env.stepsize:
            raise ValueError(\
                'windowsize and stepsize must be equal when using AIM features')
        
        self.env = Environment.instance
        self.sig = aimc.SignalBank()
        self.sig.Initialize(1,self.env.windowsize,self.env.samplerate)
        
        pzfc_params = aimc.Parameters()
        pzfc_params.SetFloat('pzfc.highest_frequency',highest_freq)
        self.pzfc = aimc.ModulePZFC(pzfc_params)
        self._init()
        self.output_bank = self.feature.GetOutputBank()
        self.global_params = aimc.Parameters()
        self.pzfc.Initialize(self.sig,self.global_params)
        self._dim = \
            (self.output_bank.channel_count(),self.output_bank.buffer_length())
        
    @property
    def channels(self):
        return self.output_bank.channel_count()
     
    def dim(self,env):
        return np.product(self._dim)
    
    @property
    def dtype(self):
        return np.float32
    
    # TODO: _process needs to be written in cython
    def _process(self):
        signal = self.in_data.astype(np.float64)
        out = np.ndarray((signal.shape[0],self.channels,signal.shape[1]),
                         dtype = np.float32)
        for i,s in enumerate(signal):
            self.sig.set_signal(0,s)
            self.pzfc.Process(self.sig)
            ob = self.feature.GetOutputBank()
            for c in range(self._dim[0]):
                out[i,c,:] = ob.get_signal(c)
        
        return flatten2d(out)


class PZFC(AIMFeature):
    
    def __init__(self,needs = None, key = None, highest_freq = 12000):
        AIMFeature.__init__(self, needs = needs, key = key, highest_freq = highest_freq)

    
    def _init(self):
        self.feature = self.pzfc
        

class NAP(AIMFeature):
    '''
    Neural Activity Pattern
    '''
    def __init__(self,needs = None, key = None, highest_freq = 12000):
        AIMFeature.__init__(self, needs = needs, key = key, highest_freq = highest_freq)
    
    
    def _init(self):
        self.hcl = aimc.ModuleHCL(aimc.Parameters())
        self.pzfc.AddTarget(self.hcl)
        self.feature = self.hcl
        

class AIM(AIMFeature):
    '''
    Auditory Image Model
    '''
    def __init__(self,needs = None, key = None, highest_freq = 12000):
        AIMFeature.__init__(self, needs = needs, key = key, highest_freq = highest_freq)
    
    def _init(self):
        self.hcl = aimc.ModuleHCL(aimc.Parameters())
        self.local_max = aimc.ModuleLocalMax(aimc.Parameters())
        self.sai = aimc.ModuleSAI(aimc.Parameters())
        self.pzfc.AddTarget(self.hcl)
        self.hcl.AddTarget(self.local_max)
        self.local_max.AddTarget(self.sai)
        self.feature = self.sai