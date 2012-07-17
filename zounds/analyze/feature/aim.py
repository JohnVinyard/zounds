import aimc
import numpy as np
from zounds.analyze.extractor import SingleInput
from zounds.environment import Environment
from zounds.util import downsample
from itertools import permutations,product

class PZFC(SingleInput):
    
    def __init__(self,needs = None, key = None):
        SingleInput.__init__(self,needs = needs, nframes = 1, step = 1, key = key)
        self.env = Environment.instance
        self.sig = aimc.SignalBank()
        self.sig.Initialize(1,self.env.windowsize,self.env.samplerate)
        
        self.pzfc = aimc.ModulePZFC(aimc.Parameters())
        self.global_params = aimc.Parameters()
        self.pzfc.Initialize(self.sig,self.global_params)
        output_bank = self.pzfc.GetOutputBank()
        self._dim = (output_bank.channel_count(),output_bank.buffer_length())
    
    def dim(self,env):
        return np.product(self._dim)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        signal = self.in_data[0].astype(np.float64)
        self.sig.set_signal(0,signal)
        self.pzfc.Process(self.sig)
        output_bank = self.pzfc.GetOutputBank()
        output = np.zeros(self._dim)
        for c in range(self._dim[0]):
            output[c] = np.array(output_bank.get_signal(c))
        
        return output.ravel()

class NAP(SingleInput):
    '''
    Neural Activity Pattern
    '''
    def __init__(self,needs = None, key = None):
        SingleInput.__init__(self,needs = needs,nframes = 1, step = 1, key = key)
        self.env = Environment.instance
        self.sig = aimc.SignalBank()
        self.sig.Initialize(1,self.env.windowsize,self.env.samplerate)
        
        pzfc_params = aimc.Parameters()
        pzfc_params.SetFloat('pzfc.highest_frequency',12000.)
        self.pzfc = aimc.ModulePZFC(pzfc_params)
        hcl_params = aimc.Parameters()
        self.hcl = aimc.ModuleHCL(hcl_params)
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
        signal = self.in_data[0].astype(np.float64)
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
        
        pzfc_params = aimc.Parameters()
        pzfc_params.SetFloat('pzfc.highest_frequency',12000.)
        self.pzfc = aimc.ModulePZFC(pzfc_params)
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
        signal = self.in_data[0].astype(np.float64)
        self.sig.set_signal(0,signal)
        self.pzfc.Process(self.sig)
        output_bank = self.sai.GetOutputBank()
        output = np.zeros(self._dim)
        for c in range(self._dim[0]):
            output[c] = np.array(output_bank.get_signal(c))
        
        return output.ravel()



# BUG: This class is totally broken!!!
class GoogleBoxCut(SingleInput):
    '''
    Box-Cutting algorithm, as discussed here 
    http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/pubs/archive/35479.pdf
    
    TODO: The calculation of _total_boxes and _total_size is wrong!
    TODO: Generalize this, so it can be used for any feature of any dimensionality
    '''
    def __init__(self,needs = None, key = None,\
                 aim_size = None,overlap = .5,\
                 smallest_box_size = None, scales = None):
        SingleInput.__init__(self, step = 1, nframes = 1, \
                             needs = needs, key = key)
        self._as = np.array(aim_size)
        self._sbs = np.array(smallest_box_size)
        self._dims = len(self._sbs)
        self._vec_size = np.sum(self._sbs)
        self._scales = np.array(scales)
        self._overlap = overlap
        
        # calculate the total number of boxes that will fit into 
        # the whole aim frame
        self._total_boxes = 0
        # keep track of the number of boxes at each scale
        self._bas = {}
        for s in self._scales:
            cs = s * self._sbs
            frames = np.floor((self._as - cs) / (cs * overlap))
            bas = np.product(frames)
            self._bas[s] = bas
            self._total_boxes += bas
        
        # calculate the size of a single box vector after doing the following:
        # 1) downsample to smallest box size
        # 2) take the average along each dimension
        # 3) concatentate the sum vectors
        las = len(self._as)
        indices = np.arange(las)
        s = list(set([frozenset(q) for q in permutations(indices,las - 1)]))
        self._total_size = 0
        for ind in s:
            dims = self._sbs[list(ind)]
            pdims = np.product(dims)
            self._total_size += pdims
        
        # the total size of this feature is the size of a single box vector
        # times the number of boxes
        self._total_size *= self._total_boxes
        self._total_size = int(self._total_size)

    def dim(self,env):
        return self._total_size
    
    @property
    def dtype(self):
        return np.float32
    
    _PRODUCT_CACHE = {}
    def _product(self,scale,data):
        try:
            return GoogleBoxCut._PRODUCT_CACHE[scale]
        except KeyError:
            shape = self._sbs * scale
            slices = [[slice(i,i + shape[j]) 
                   for i in xrange(0,data.shape[j] - shape[j],shape[j] * self._overlap)] 
                  for j in xrange(len(shape))]
            prod = list(product(*slices))
            GoogleBoxCut._PRODUCT_CACHE[scale] = prod
            return prod
            
    
    def _boxes_at_scale(self,scale,data):
        boxes = np.zeros((self._bas[scale],self._vec_size))
        prod = self._product(scale, data)
        for i,coords in enumerate(prod):
            box = downsample(data[coords],scale)
            vec = np.concatenate([box.mean(i).ravel() for i in xrange(self._dims)])
            boxes[i] = vec
        return boxes.ravel()
    
    def _process(self):
        # KLUDGE: This will only work for nframes = 1
        data = self.in_data[0].reshape(self._as)
        data = np.concatenate([self._boxes_at_scale(s, data) for s in self._scales])
        return data
        