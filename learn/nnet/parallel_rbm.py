from rbm import Rbm
from multiprocessing.sharedctypes import RawArray
from multiprocessing import Pool
import ctypes
import numpy as np

_ctypes = [ctypes.c_char, 
           ctypes.c_wchar, 
           ctypes.c_byte, 
           ctypes.c_ubyte, 
           ctypes.c_short, 
           ctypes.c_ushort, 
           ctypes.c_int,
           ctypes.c_uint, 
           ctypes.c_long, 
           ctypes.c_ulong,  
           ctypes.c_float, 
           ctypes.c_double]

_np_types = [np.int8,
             np.int16,
             np.int8,
             np.uint8, 
             np.int16, 
             np.uint16, 
             np.int32, 
             np.int32, 
             np.int32, 
             np.int32,  
             np.float32, 
             np.float64]

_type_codes = ['c',
               'c',
               'B',
               'b',
               'h',
               'H',
               'i',
               'I',
               'l',
               'L',
               'f',
               'd']

def _np_to_ctype(np_type):
    '''
    Given a numpy dtype, return the equivalent ctype
    '''
    return _ctypes[_np_types.index(np_type)]

def _ctype_to_np(c_type):
    '''
    Given a ctype, return the equivalent numpy dtype
    '''
    return _np_types[_ctypes.index(c_type)]

def _typecode_from_np_type(np_type):
    '''
    Return a typecode string, given a numpy dtype
    '''
    return _type_codes[_np_types.index(np_type)]

def _typecode_from_ctype(c_type):
    '''
    Return a typecode string, given a ctype
    '''
    return _type_codes[_ctypes.index(c_type)]


def ndarray_as_shmem(ndarray):
    size = ndarray.size
    typecode = _typecode_from_np_type(ndarray.dtype)
    ra = RawArray(typecode,size)
    ra[:] = ndarray.tostring()
    return ra

def shmem_as_ndarray( raw_array ):
    address = raw_array._wrapper.get_address()     
    size = raw_array._wrapper.get_size() 
    dtype = _ctype_to_np(raw_array._type_)
    class Dummy(object): pass 
    d = Dummy() 
    d.__array_interface__ = { 
         'data' : (address, False), 
         'typestr' : np.uint8.str, 
         'descr' : np.uint8.descr, 
         'shape' : (size,), 
         'strides' : None, 
         'version' : 3 
    }     
    return np.asarray(d).view( dtype=dtype )

def update(
           # shared memory representing weights and biases
           sh_weights,
           sh_vbias,
           sh_hbias,
           # shared memory representing velocities for use
           # during training
           sh_wvelocity,
           sh_vbvelocity,
           sh_hbvelocity,
           # shared memory representing updates, e.g., for four
           # processes, there should be four weight update matrices
           sh_sparsity_updates,
           sh_wvelocity_updates,
           sh_vbvelocity_updates,
           sh_hbvelocity_updates,
           # this worker's index. This is how we know where to write in
           # the update matrices
           worker_index,
           rbm,
           # input data and info
           sh_inp,
           batch_size,
           # training stuff
           momentum,
           epoch,
           batch):
    
    indim = rbm.indim
    hdim = rbm.hdim
    
    # view shared memory as numpy arrays
    weights = shmem_as_ndarray(sh_weights).reshape((indim,hdim))
    vbias = shmem_as_ndarray(sh_vbias)
    hbias = shmem_as_ndarray(sh_hbias)
    wvelocity = shmem_as_ndarray(sh_wvelocity).reshape((indim,hdim))
    vbvelocity = shmem_as_ndarray(sh_vbvelocity)
    hbvelocity = shmem_as_ndarray(sh_hbvelocity)
    inp = shmem_as_ndarray(sh_inp).reshape((batch_size,indim))
    
    rbm.set_data(weights,hbias,vbias)
    stoch, posprod, pos_h_act, pos_v_act = rbm._positive_phase(inp)
    v,negprod,neg_h_act,neg_v_act = rbm._negative_phase(stoch)
    error = np.sum(np.abs(inp - v) ** 2)
    n = batch_size
    m = momentum
    lr = rbm._learning_rate
    wd = rbm._weight_decay
    
    # sparsity updates
    sparsity_update = None
    if None != rbm._sparsity_target:
        current_sparsity = stoch.sum(0) / float(n)
        sparsity_update = (rbm._sparsity_decay * rbm._sparsity) + \
            ((1 - rbm._sparsity_decay) * current_sparsity)
        sparse_penalty = rbm._sparsity_cost * \
            (rbm._sparsity_target - sparsity_update)
    sparsity_start = hdim*worker_index
    sparsity_stop = sparsity_start + hdim
    sh_sparsity_updates[sparsity_start : sparsity_stop] =\
         sparsity_update.tostring()
    
    # weight updates
    wvelocity_update = (m * rbm._wvelocity) + \
            lr * (((posprod-negprod)/n) - (wd*weights))
    if None is not rbm.sparsity_target:
        wvelocity_update += sparse_penalty
    w_size = weights.size
    w_start = w_size * worker_index
    w_stop = w_start + w_size
    sh_wvelocity_updates[w_start : w_stop] = wvelocity_update.tostring()
    
    # visual bias updates
    vbvelocity_update =  (m * rbm._vbvelocity) + \
            ((lr/n) * (pos_v_act - neg_v_act))
    vb_start = indim * worker_index
    vb_stop = vb_start + indim
    sh_vbvelocity_updates[vb_start : vb_stop] = vbvelocity_update.tostring()
    
    # hidden bias updates
    hbvelocity_update = (m * rbm._hbvelocity) + \
            ((lr/n) * (pos_h_act - neg_h_act))
    if None is not rbm._sparsity_target:
        hbvelocity_update += sparse_penalty
    hb_start = hdim * worker_index
    hb_stop = hb_start + hdim
    sh_hbvelocity_updates[hb_start : hb_stop] = hbvelocity.tostring()
    
    return error
    
class RbmShim(Rbm):
    '''
    A class that knows how to do the work of an rbm, but doesn't have all the 
    memory-intensive objects attached, like weights and biases
    '''
    def __init__(self,indim,hdim):
        Rbm.__init__(self,indim,hdim)
        self._weights = None
        self._hbias = None
        self._vbias = None
    
    
    def set_data(self,weights,hbias,vbias):
        self._weights = weights
        self._hbias = hbias
        self._vbias = vbias

class ParallelRbm(Rbm):
    '''
    A binary-binary rbm that trains in parallel, processing multiple mini-batches
    at once
    '''
    def __init__(self,indim,hdim,nworkers = 4):
        Rbm.__init__(self,indim,hdim)
        self.nworkers = nworkers
    
    def train(self,samples,stopping_condition):
        batch_size = 100
        nbatches = len(samples) / batch_size
        samples = samples.reshape((nbatches,batch_size,samples.shape[1]))
        shim = RbmShim(self.indim,self.hdim)
        
        self._wvelocity = np.zeros(self._weights.shape)
        self._vbvelocity = np.zeros(self._indim)
        self._hbvelocity = np.zeros(self._hdim)
        self._sparsity = np.zeros(self._hdim)
        
        weights = ndarray_as_shmem(self._weights)
        wvelocity = ndarray_as_shmem(self._wvelocity)
        wvelocity_updates = RawArray('d',self._weights.size * self.nworkers)
        
        vbias = ndarray_as_shmem(self._vbias)
        vbvelocity = ndarray_as_shmem(self._vbvelocity)
        vbvelocity_updates = RawArray('d',self._vbias.size * self.nworkers)
        
        hbias = ndarray_as_shmem(self._hbias)
        hbvelocity = ndarray_as_shmem(self._hbvelocity)
        hbvelocity_updates = RawArray('d',self._hbias.size * self.nworkers)
        
        sparsity_updates = RawArray('d',self.hdim * self.nworkers)
        
        sh_samples = ndarray_as_shmem(samples)
        
        epoch = 0
        error = [9999999] * self.nworkers
        nbatches = len(samples)
        
        while not any([stopping_condition(epoch,e) for e in error]):
            if epoch < 5:
                mom = self._initial_momentum
            else:
                mom = self._final_momentum
            batch = 0
            while batch < nbatches and\
             not any([stopping_condition(epoch,e) for e in error]):
                pool = Pool(self.nworkers)
                '''
                    # shared memory representing weights and biases
                   sh_weights,
                   sh_vbias,
                   sh_hbias,
                   # shared memory representing velocities for use
                   # during training
                   sh_wvelocity,
                   sh_vbvelocity,
                   sh_hbvelocity,
                   # shared memory representing updates, e.g., for four
                   # processes, there should be four weight update matrices
                   sh_sparsity_updates,
                   sh_wvelocity_updates,
                   sh_vbvelocity_updates,
                   sh_hbvelocity_updates,
                   # this worker's index. This is how we know where to write in
                   # the update matrices
                   worker_index,
                   rbm,
                   # input data and info
                   sh_inp,
                   batch_size,
                   # training stuff
                   momentum,
                   epoch,
                   batch):
               '''
                data = []
                for i in self.nworkers:
                    data.append((weights,vbias,hbias,wvelocity,vbvelocity,
                                 hbvelocity,sparsity_updates,wvelocity_updates,
                                 vbvelocity_updates,hbvelocity_updates,i,
                                 shim,????,batch_size,mom,epoch,batch))
                    errs = pool.apply(update,data)
                    # TODO: print errors
                    # TODO: do updates
                    
                
                
        
        # get rid of the "temp" training variables
        del self._wvelocity
        del self._vbvelocity
        del self._hbvelocity
        del self._sparsity
        