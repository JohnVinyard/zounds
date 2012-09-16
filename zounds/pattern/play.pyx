cimport numpy as np
import numpy as np

FLOAT_DTYPE = np.float32
ctypedef np.float32_t FLOAT_DTYPE_t

UINT64_DTYPE = np.uint64
ctypedef np.uint64_t UINT64_DTYPE_t


cdef extern from 'cplay.h':
    void setup()
    void teardown()
    void put_event(float *buf,unsigned int start_sample,unsigned int stop_sample, UINT64_DTYPE_t start_time_ms,char done)
    void cancel_all_events()
    UINT64_DTYPE_t get_time()
    
def start():
    setup()

def stop():
    teardown()

def usecs():
    return get_time();

def put(np.ndarray[FLOAT_DTYPE_t,ndim = 1] n,int starts, int stops,UINT64_DTYPE_t time):
    put_event(<float*>n.data,starts,stops,time,0)

def cancel_all():
    cancel_all_events()
    
    