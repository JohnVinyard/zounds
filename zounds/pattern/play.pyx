cimport numpy as np
import numpy as np

FLOAT_DTYPE = np.float32
ctypedef np.float32_t FLOAT_DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT32_DTYPE_t

UINT64_DTYPE = np.uint64
ctypedef np.uint64_t UINT64_DTYPE_t

from libc.stdlib cimport malloc, free

cdef extern from 'cplay.h':
    void setup()
    void teardown()
    void cancel_all_events()
    UINT64_DTYPE_t get_time()
    
    
    # Parameter ###############################################################
    ctypedef struct parameter:
        pass
    
    parameter * parameter_new(
        float * values,int n_values,UINT64_DTYPE_t * times,char * interpolations)
    
    # Transform ###############################################################
    ctypedef struct transform:
        pass
    
    transform * gain_new(parameter * params)
    transform * delay_new(int max_delay_time,parameter * params)
    
    # Event ###############################################################
    ctypedef struct event2:
        pass
    
    event2 * event2_new_leaf(
        float * buf,int start_sample,int stop_sample,jack_nframes_t start_time,
        char unknown_length,transform * transforms,int n_transforms)
    
    event2 * event2_new_branch(
        event2 * children,int n_children,jack_nframes_t start_time,
        char unknown_length,transform * transforms,int n_transforms)
    
def start():
    setup()

def stop():
    teardown()

def usecs():
    '''
    Get the current time in microseconds, according to JACK
    '''
    return get_time();

#def put(np.ndarray[FLOAT_DTYPE_t,ndim = 1] n,int starts, int stops,UINT64_DTYPE_t time):
#    put_event(<float*>n.data,starts,stops,time,0)

def cancel_all():
    '''
    Cancel all pending events
    '''
    cancel_all_events()


def build_event_and_enqueue(e, patterns, buffers, k, time, event2 event = NULL):
    '''
    Recursively build event and enqueue it
    '''
    ptrn = patterns[k]
    # build transforms
    

def put2(ptrn, patterns, buffers, start_time):
    '''
    Add a pattern to the queue
    '''
    for k,v in ptrn.pdata.iteritems():
        for e in v:
            # TODO: Build each event recursively and enqueue it 
            build_event_and_enqueue(e,patterns, buffers, k, start_time)
    
    
    
    