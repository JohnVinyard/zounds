from rbm import Rbm,LinearRbm
from multiprocessing.sharedctypes import Array,RawArray
from multiprocessing import Process,Pool
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
_np_type_strs = dict([(dt,np.zeros(0,dtype=dt).dtype.str) for dt in _np_types])
_np_type_descrs = dict([(dt,np.zeros(0,dtype=dt).dtype.descr) for dt in _np_types])

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
    typecode = _typecode_from_np_type(ndarray.dtype)
    ra = RawArray(typecode,ndarray.reshape(ndarray.size))
    return ra

def shmem_as_ndarray( raw_array ):
    address = raw_array._wrapper.get_address()      
    dtype = _ctype_to_np(raw_array._type_)
    class Dummy(object): pass 
    d = Dummy() 
    d.__array_interface__ = { 
         'data' : (address, False), 
         'typestr' : _np_type_strs[dtype], 
         'descr' : _np_type_descrs[dtype], 
         'shape' : (len(raw_array),), 
         'strides' : None, 
         'version' : 3 
    }     
    return np.asarray(d).view( dtype=dtype )

def update(
           # shared memory representing weights and biases
           sh_weights,
           sh_vbias,
           sh_hbias,
           sh_sparsity,
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
           nbatches,
           # training stuff
           momentum,
           epoch,
           batch):
    
    indim = rbm._indim
    hdim = rbm._hdim
    
    # view shared memory as numpy arrays
    weights = shmem_as_ndarray(sh_weights).reshape((indim,hdim))
    vbias = shmem_as_ndarray(sh_vbias)
    hbias = shmem_as_ndarray(sh_hbias)
    wvelocity = shmem_as_ndarray(sh_wvelocity).reshape((indim,hdim))
    vbvelocity = shmem_as_ndarray(sh_vbvelocity)
    hbvelocity = shmem_as_ndarray(sh_hbvelocity)
    sparsity = shmem_as_ndarray(sh_sparsity)
    inp = shmem_as_ndarray(sh_inp).reshape((nbatches,batch_size,indim))[batch]
    
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
        sparsity_update = (rbm._sparsity_decay * sparsity) + \
            ((1 - rbm._sparsity_decay) * current_sparsity)
        sparse_penalty = rbm._sparsity_cost * \
            (rbm._sparsity_target - sparsity_update)
    sparsity_start = hdim*worker_index
    sparsity_stop = sparsity_start + hdim
    sh_sparsity_updates[sparsity_start : sparsity_stop] =sparsity_update
    
    # weight updates
    wvelocity_update = (m * wvelocity) + \
            lr * (((posprod-negprod)/n) - (wd*weights))
    if None is not rbm._sparsity_target:
        wvelocity_update += sparse_penalty
    w_size = weights.size
    w_start = w_size * worker_index
    w_stop = w_start + w_size 
    sh_wvelocity_updates[w_start : w_stop] = wvelocity_update.reshape(w_size)
    
    # visual bias updates
    vbvelocity_update =  (m * vbvelocity) + \
            ((lr/n) * (pos_v_act - neg_v_act))
    vb_start = indim * worker_index
    vb_stop = vb_start + indim
    sh_vbvelocity_updates[vb_start : vb_stop] = vbvelocity_update
    
    # hidden bias updates
    hbvelocity_update = (m * hbvelocity) + \
            ((lr/n) * (pos_h_act - neg_h_act))
    if None is not rbm._sparsity_target:
        hbvelocity_update += sparse_penalty
    hb_start = hdim * worker_index
    hb_stop = hb_start + hdim
    print hbvelocity_update.shape
    print len(sh_hbvelocity_updates)
    print hb_start - hb_stop
    sh_hbvelocity_updates[hb_start : hb_stop] = hbvelocity_update
    
    print 'Epoch %i, Batch %i, Error %1.4f' % (epoch,batch,error)
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
        print 'Parallel training'
        batch_size = 100
        nbatches = len(samples) / batch_size
        sample_size = samples.shape[1]
        samples = samples.reshape((nbatches,batch_size,sample_size))
        shim = RbmShim(self._indim,self._hdim)
        
        self._wvelocity = np.zeros(self._weights.shape)
        self._vbvelocity = np.zeros(self._indim)
        self._hbvelocity = np.zeros(self._hdim)
        self._sparsity = np.zeros(self._hdim)
        
        weights = ndarray_as_shmem(self._weights)
        self._weights = shmem_as_ndarray(weights).reshape(self._weights.shape)
         
        wvelocity = ndarray_as_shmem(self._wvelocity)
        self._wvelocity = shmem_as_ndarray(wvelocity).reshape(self._wvelocity.shape)
        
        wvelocity_updates = RawArray('d',self._weights.size * self.nworkers)
        self._wvelocity_updates = shmem_as_ndarray(wvelocity_updates)
        
        vbias = ndarray_as_shmem(self._vbias)
        self._vbias = shmem_as_ndarray(vbias).reshape(self._vbias.shape)
        
        vbvelocity = ndarray_as_shmem(self._vbvelocity)
        self._vbvelocity = shmem_as_ndarray(vbvelocity).reshape(self._vbvelocity.shape)
        
        vbvelocity_updates = RawArray('d',self._vbias.size * self.nworkers)
        self._vbvelocity_updates = shmem_as_ndarray(vbvelocity_updates)
        
        print self._hbias.shape
        hbias = ndarray_as_shmem(self._hbias)
        print len(hbias)
        self._hbias = shmem_as_ndarray(hbias).reshape(self._hbias.shape)
        
        hbvelocity = ndarray_as_shmem(self._hbvelocity)
        self._hbvelocity = shmem_as_ndarray(hbvelocity).reshape(self._hbvelocity.shape)
        
        hbvelocity_updates = RawArray('d',self._hbias.size * self.nworkers)
        print len(hbvelocity_updates)
        self._hbvelocity_updates = shmem_as_ndarray(hbvelocity_updates)
        
        sh_sparsity = RawArray('d',self._hdim)
        self._sparsity = shmem_as_ndarray(sh_sparsity).reshape(self._sparsity.shape)
        
        sparsity_updates = RawArray('d',self._hdim * self.nworkers)
        self._sparsity_updates = shmem_as_ndarray(sparsity_updates)
        
        sh_samples = ndarray_as_shmem(samples)
        sh_batch_size = batch_size * sample_size
        
        epoch = 0
        error = [9999999] * self.nworkers
        nbatches = len(samples)
        
        def create_process(args):
            return Process(target = update,args = args)
        
        while not any([stopping_condition(epoch,e) for e in error]):
            if epoch < 5:
                mom = self._initial_momentum
            else:
                mom = self._final_momentum
            batch = 0
            while batch < nbatches and\
             not any([stopping_condition(epoch,e) for e in error]):
                data = []
                for i in range(self.nworkers):
                    offset = sh_batch_size * batch
                    '''
                   # shared memory representing weights and biases
           sh_weights,
           sh_vbias,
           sh_hbias,
           sh_sparsity,
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
           nbatches,
           # training stuff
           momentum,
           epoch,
           batch):'''
                    
                    data.append((weights,
                                vbias,
                                hbias,
                                sh_sparsity,
                                wvelocity,
                                vbvelocity,
                                hbvelocity,
                                sparsity_updates,
                                wvelocity_updates,
                                vbvelocity_updates,
                                hbvelocity_updates,
                                i,
                                shim,
                                sh_samples,
                                batch_size,
                                nbatches,
                                mom,
                                epoch,
                                batch + i))
                    
                procs = map(create_process, data)
                map(lambda p : p.start(), procs)
                map(lambda p: p.join(), procs)
                batch += self.nworkers
                
                # Do updates
                # average the sparsity updates to get the new sparsity
                self._sparsity = np.average(\
                        self._sparsity_updates.reshape((self.nworkers,self._hdim)),0)
                self._wvelocity = np.average(\
                        self._wvelocity_updates.reshape(self.nworkers,self._hdim),0)
                self._vbvelocity = np.average(\
                        self._vbvelocity_updates.reshape(self.nworkers,self._indim),0)
                self._hbvelocity = np.average(\
                        self._hbvelocity_updates.reshape(self.nworkers,self._hdim),0)
                for i in range(len(self.nworkers)):
                    self._weights += \
                        self.wvelocity_updates\
                        [i * self._hdim : i * self._hdim + self._hdim]
                    self._vbias += \
                        self._vbvelocity_updates\
                        [i * self._indim : i * self._indim + self._indim]
                    self._hbias += \
                        self._hbvelocity_updates\
                        [i * self._hdim : i * self._hdim + self._hdim]
                    
                    
                    
                    
                
                
                    
                
                
        
        # get rid of the "temp" training variables
        del self._wvelocity
        del self._vbvelocity
        del self._hbvelocity
        del self._sparsity

class ParallelLinearRbm(ParallelRbm,LinearRbm):
    def __init__(self,indim,hdim,nworkers = 4):
        LinearRbm.__init__(self,indim,hdim)
        ParallelRbm.__init__(self,indim,hdim,nworkers = nworkers)
        

if __name__ == '__main__':
    '''
    print 'np.ndarray -> shared memory doesn\'t work'
    a = np.zeros(10)
    ra = ndarray_as_shmem(a)
    
    a[:] = 100
    print a
    print ra[:]
    ra[0] = 999
    print a
    print ra[:]
    
    del a
    del ra
    
    print 'shared memory -> np.ndarray does'
    ra = RawArray('d',10)
    a = shmem_as_ndarray(ra)
    ra[0] = 100
    print a
    print ra[:]
    a[1] = 999
    print a
    print ra[:]
    '''
    
    def s(arr,workern):
        arr[workern] = workern
        return True
    
    def create_process(args):
        #arr,workern = args
        return Process(target = s, args = args)
    
    ra = RawArray('d',4)
    procs = map(create_process, [(ra,i) for i in range(4)])
    map(lambda p : p.start(), procs)
    map(lambda p: p.join(), procs)
    print ra[:]
    print [proc.result for proc in procs]
    
    
    