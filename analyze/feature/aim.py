import aimc
import numpy as np
from analyze.extractor import SingleInput
from environment import Environment
from util import downsample,downsample3d
from itertools import permutations,product

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
        self.global_params.SetBool('nap.do_log_compression',True)
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


class GoogleBoxCut(SingleInput):
    '''
    Box-Cutting algorithm, as discussed here 
    http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/pubs/archive/35479.pdf
    
    TODO: The calculation of _total_boxes and _total_size is wrong!
    TODO: Generalize this, so it can be used for any feature of any dimensionality
    '''
    def __init__(self,needs = None, key = None, nframes = 1,step = 1,\
                 aim_size = None,overlap = .5,\
                 smallest_box_size = None, scales = None):
        SingleInput.__init__(self, step = step, nframes = nframes, \
                             needs = needs, key = key)
        self._as = np.array(aim_size)
        self._sbs = np.array(smallest_box_size)
        self._dims = len(self._sbs)
        self._vec_size = np.product(self._sbs)
        self._scales = np.array(scales)
        self._overlap = overlap
        
        # calculate the total number of boxes that will fit into 
        # the whole aim frame
        self._total_boxes = 0
        for s in self._scales:
            cs = s * self._sbs
            frames = np.floor((self._as - cs) / (cs * overlap))
            self._total_boxes += np.product(frames)
        
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
        
        self._downsample = self._downsample2d if 2 == self._dims \
                            else self._downsample3d

    def dim(self,env):
        return self._total_size
    
    @property
    def dtype(self):
        return np.float32
    
    def _downsample2d(self,arr,factor):
        return downsample(arr,factor)
    
    def _downsample3d(self,arr,factor):
        return downsample3d(arr,factor)
    
    def _boxes_at_scale(self,scale,data):
        shape = self._sbs * scale
        slices = [[slice(i,i + shape[j]) 
                   for i in range(0,data.shape[j] - shape[j],shape[j] * self._overlap)] 
                  for j in range(len(shape))]
        boxes = []
        for coords in product(*slices):
            box = self._downsample(data[coords],scale)
            vec = np.concatenate([box.mean(i).ravel() for i in range(self._dims)])
            boxes.append(vec)
        print len(boxes)
        return np.array(boxes).ravel()
    
    def _process(self):
        data = np.array(self.in_data)
        data = data.reshape(self._as)
        data = np.concatenate([self._boxes_at_scale(s, data) for s in self._scales])
        print data.shape
        return data
        