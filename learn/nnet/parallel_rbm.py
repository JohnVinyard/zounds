from rbm import Rbm
from nnet import stochastic_binary as sb,sigmoid
from multiprocessing.sharedctypes import Array,RawArray
from multiprocessing import Process
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


def cheatdot(sh_buf,samples,weights,i,chunksize):
    start = i * chunksize
    stop = start + chunksize 
    r = np.dot(samples[start : stop],weights)
    dim = weights.shape[1]
    sh_buf[start * dim: stop * dim] = r.reshape(r.size)
    
def create_process(args):
    return Process(target = cheatdot, args = args)
        
def pdot(out,samples,weights,nprocesses,chunksize): 
    procs = map(create_process,
                [(out,samples,weights,i,chunksize) for i in xrange(nprocesses)])
    map(lambda p : p.start(), procs)
    map(lambda p: p.join(), procs)

class ParallelRbm2(Rbm):
    
    def __init__(self,
                 indim,
                 hdim,
                 learning_rate = 0.1,
                 weight_decay = .002,
                 initial_momentum = .5,
                 final_momentum = .9,
                 sparsity_target = .01,
                 sparsity_decay = .9,
                 sparsity_cost = .001):
        Rbm.__init__(self,
                     indim,
                     hdim,
                     learning_rate = learning_rate,
                     weight_decay = weight_decay,
                     initial_momentum = initial_momentum,
                     final_momentum = final_momentum,
                     sparsity_target = sparsity_target,
                     sparsity_decay = sparsity_decay,
                     sparsity_cost = sparsity_cost)
        
        self.batch_size = 100
        self.nprocesses = 2
        self.chunksize = self.batch_size / self.nprocesses
        # BUG: What if indim isn't evenly divisible by nprocesses?
        self.phase_chunksize = self._indim / self.nprocesses
        
        
        
        
    
    def _up(self,v):
        pdot(self._sh_up_pre_sigmoid,v,self._weights,self.nprocesses,self.chunksize)
        self._up_pre_sigmoid += self._hbias
        return self._up_pre_sigmoid,sigmoid(self._up_pre_sigmoid)
    
    def _down(self,h):
        pdot(self._sh_down_pre_sigmoid,h,self._weights.T,self.nprocesses,self.chunksize)
        self._down_pre_sigmoid += self._vbias
        sig = sigmoid(self._down_pre_sigmoid)
        return self._down_pre_sigmoid,sig,sb(sig)
    
    def _positive_phase(self,inp):
        ps,s,stoch = self._h_from_v(inp)
        #pdot(self._sh_posprod,inp.T,s,self.nprocesses,self.phase_chunksize)
        self._posprod = np.dot(inp.T,s)
        pos_h_act = s.sum(axis = 0)
        pos_v_act = inp.sum(axis = 0)
        return stoch, self._posprod, pos_h_act, pos_v_act
    
    def _negative_phase(self,stoch):
        vs,gps,gs,gstoch = self._gibbs_hvh(stoch)
        #pdot(self._sh_negprod,vs.T,gs,self.nprocesses,self.phase_chunksize)
        self._negprod = np.dot(vs.T,gs)
        neg_h_act = gs.sum(axis = 0)
        neg_v_act = vs.sum(axis = 0)
        return vs, self._negprod, neg_h_act, neg_v_act
    
    
    def train(self,samples,stopping_condition):
        
        self._sh_samples = Array(ctypes.c_double,samples.size)
        self._sh_samples[:] = samples.reshape(samples.size)
        self._samples = np.ctypeslib.as_array(self._sh_samples.get_obj()).reshape(samples.shape)
        
        self._up_pre_sigmoid =  np.zeros((self.batch_size,self._hdim))
        self._sh_up_pre_sigmoid = ndarray_as_shmem(self._up_pre_sigmoid)
        self._up_pre_sigmoid =\
            shmem_as_ndarray(self._sh_up_pre_sigmoid)\
            .reshape(self._up_pre_sigmoid.shape)
        
        
        self._down_pre_sigmoid = np.zeros((self.batch_size,self._indim))
        self._sh_down_pre_sigmoid = ndarray_as_shmem(self._down_pre_sigmoid)
        self._down_pre_sigmoid =\
            shmem_as_ndarray(self._sh_down_pre_sigmoid)\
            .reshape(self._down_pre_sigmoid.shape)
        
            
        self._posprod = np.zeros((self._indim,self._hdim))
        self._sh_posprod = ndarray_as_shmem(self._posprod)
        self._posprod =\
             shmem_as_ndarray(self._sh_posprod).reshape(self._posprod.shape)
        
        self._negprod = np.zeros((self._indim,self._hdim))
        self._sh_negprod = ndarray_as_shmem(self._negprod)
        self._negprod =\
            shmem_as_ndarray(self._sh_negprod).reshape(self._negprod.shape)
            
        self._sh_weights = Array(ctypes.c_double,self._indim * self._hdim)
        self._sh_weights[:] = self._weights.reshape(self._weights.size)
        self._weights = np.ctypeslib.as_array(self._sh_weights.get_obj()).reshape(self._weights.shape)
        
        Rbm.train(self,self._samples,stopping_condition)
        
        del self._up_pre_sigmoid
        del self._sh_up_pre_sigmoid
        del self._down_pre_sigmoid
        del self._sh_down_pre_sigmoid
        del self._posprod
        del self._sh_posprod
        del self._negprod
        del self._sh_negprod
        del self._sh_weights
        del self._sh_samples
        del self._samples
        

class ParallelLinearRbm2(ParallelRbm2):
    
    def __init__(self,indim,hdim,sparsity_target = .01, learning_rate = .001):
        ParallelRbm2.__init__(self,
                              indim,
                              hdim,
                              sparsity_target = sparsity_target,
                              learning_rate = learning_rate)
    
    def _gibbs_hvh(self,h):
        '''
        One step of gibbs sampling, starting from the
        hidden layer
        '''
        vps,vs,vstoch = self._v_from_h(h)
        # use the actual value of the visible units,
        # instead of the value passed through a sigmoid function
        ps,s,stoch = self._h_from_v(vps)
        return vps,ps,s,stoch

    def activate(self,inp):
        hps,hs,hstoch = self._h_from_v(inp)
        vps,vs,vstoch = self._v_from_h(hs)
        return vps

from time import time

if __name__ == '__main__':
    
    
    indim = 500
    hdim = 2000
    itr = 50
    nsamples = 100
    weights = np.random.random_sample((indim,hdim))
    samples = np.random.random_sample((nsamples,indim))
    
    start = time()
    for i in range(itr):
        np.dot(samples,weights)
    print 'np.dot took %1.4f' % (time() - start)
    
    
    buf = np.zeros((nsamples,hdim))
    sh_buf = ndarray_as_shmem(buf)
    buf = shmem_as_ndarray(sh_buf).reshape(buf.shape)
    
    nprocesses = 2
    chunksize = nsamples / nprocesses
    
    
    
    def cheatdot(sh_buf,samples,weights,i):
        start = i * chunksize
        stop = start + chunksize 
        r = np.dot(samples[start : stop],weights)
        sh_buf[start * hdim: stop * hdim] = r.reshape(r.size)
    
    def create_process(args):
        return Process(target = cheatdot, args = args)
    
    def pdot(): 
        procs = map(create_process,[(sh_buf,samples,weights,i) for i in range(nprocesses)])
        map(lambda p : p.start(), procs)
        map(lambda p: p.join(), procs)
    
    start = time()
    for i in range(itr):
        pdot()
    print 'pdot took %1.4f' % (time() - start)
    
    r = np.dot(samples,weights)
    print np.allclose(buf,r)
        
    
    
    