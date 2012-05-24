import aimc
import numpy as np
from analyze.extractor import SingleInput
from environment import Environment


class NAP(SingleInput):
    '''
    Neural Activity Pattern
    '''
    def __init__(self,needs = None, key = None):
        SingleInput.__init__(self,needs = needs,nframes = 1, step = 1, key = key)
        self.env = Environment.instance
        self.sig = aimc.SignalBank()
        self.sig.Initialize(1,self.env.windowsize,self.env.samplerate)
        
        self.pzfc = aimc.ModulePZFC(aimc.Parameters())
        self.hcl = aimc.ModuleHCL(aimc.Parameters())
        self.pzfc.AddTarget(self.hcl)
        self.global_params = aimc.Parameters()
        self.pzfc.Initialize(self.sig,self.global_params)
        
        output_bank = self.hcl.GetOutputBank()
        
        self._dim = (output_bank.channel_count(),output_bank.buffer_length())
         
    
    def dim(self,env):
        return np.product(self._dim)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        signal = self.in_data[0]
        self.sig.set_signal(0,signal)
        self.pzfc.Process(self.sig)
        output_bank = self.hcl.GetOutputBank()
        output = np.zeros(self._dim)
        for c in range(self._dim[0]):
            output[c] = np.array(output_bank.get_signal(c))
        
        return output.ravel()
    
    
    
class AIM(SingleInput):
    '''
    Auditory Image Model
    '''
    def __init__(self,needs = None, key = None):
        SingleInput.__init__(self,needs = needs,nframes = 1, step = 1, key = key)
        self.env = Environment.instance
        self.sig = aimc.SignalBank()
        self.sig.Initialize(1,self.env.windowsize,self.env.samplerate)
        
        self.pzfc = aimc.ModulePZFC(aimc.Parameters())
        self.hcl = aimc.ModuleHCL(aimc.Parameters())
        self.local_max = aimc.ModuleLocalMax(aimc.Parameters())
        self.sai = aimc.ModuleSAI(aimc.Parameters())
        
        self.pzfc.AddTarget(self.hcl)
        self.hcl.AddTarget(self.local_max)
        self.local_max.AddTarget(self.sai)
        
        self.global_params = aimc.Parameters()
        self.pzfc.Initialize(self.sig,self.global_params)
        
        output_bank = self.sai.GetOutputBank()
        
        self._dim = (output_bank.channel_count(),output_bank.buffer_length())
         
    
    def dim(self,env):
        return np.product(self._dim)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        signal = self.in_data[0]
        self.sig.set_signal(0,signal)
        self.pzfc.Process(self.sig)
        output_bank = self.sai.GetOutputBank()
        output = np.zeros(self._dim)
        for c in range(self._dim[0]):
            output[c] = np.array(output_bank.get_signal(c))
        
        return output.ravel()