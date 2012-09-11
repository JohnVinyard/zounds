cimport numpy as np
import numpy as np

FLOAT_DTYPE = np.float32
ctypedef np.float32_t FLOAT_DTYPE_t

UINT64_DTYPE = np.uint64
ctypedef np.uint64_t UINT64_DTYPE_t

cdef extern from 'cplay.h':
    void setup()
    void teardown()
    void put_event(float *buf,int start_sample,int stop_sample, UINT64_DTYPE_t start_time_ms,char done)
    UINT64_DTYPE_t get_time()
    
def start():
    setup()

def usecs():
    return get_time();

def put(np.ndarray[FLOAT_DTYPE_t,ndim = 1] n,int starts, int stops,UINT64_DTYPE_t time):
    put_event(<float*>n.data,starts,stops,time,0)
    
    