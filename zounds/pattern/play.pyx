cimport numpy as np
import numpy as np

CHAR_DTYPE = np.uint8
ctypedef np.uint8_t CHAR_DTYPE_t

FLOAT_DTYPE = np.float32
ctypedef np.float32_t FLOAT_DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT32_DTYPE_t

UINT64_DTYPE = np.uint64
ctypedef np.uint64_t UINT64_DTYPE_t

from libc.stdlib cimport malloc, free

cdef extern from 'cplay.h':
    void init_events()
    void setup()
    void teardown()
    void cancel_all_events()
    void put_event( \
        float * buf,
        unsigned int start_sample,
        unsigned int stop_sample,
        UINT64_DTYPE_t start_time_ms,
        char done)
    void put_event2(event2 * e)
    UINT64_DTYPE_t get_time()
    UINT64_DTYPE_t get_frame_time()
    
    
    # Parameter ###############################################################
    ctypedef struct parameter:
        pass
    
    void parameter_init( \
        parameter * param,float * values,int n_values,
        UINT64_DTYPE_t * times,char * interpolations)
    
    # Transform ###############################################################
    ctypedef struct transform:
        pass
    
    void gain_init(transform * t,parameter * params)
    void delay_init(transform * t,int max_delay_time,parameter * params)
    
    # Event ###############################################################
    ctypedef struct event2:
        UINT64_DTYPE_t start_time_frames
    
    event2 * event2_new_buffer(\
        float * buf,int start_sample,int stop_sample,UINT64_DTYPE_t start_time)
    
    void event2_new_base( \
        event2 * e,transform * transforms, int n_transforms,UINT64_DTYPE_t start_time_frames)
    
    void event2_new_leaf( \
        event2 * e, float * buf,int start_sample,int stop_sample,UINT64_DTYPE_t start_time,
        transform * transforms,int n_transforms)
    
    event2 * event2_new_branch( \
        event2 * children,int n_children,UINT64_DTYPE_t start_time,
        transform * transforms,int n_transforms)
    
    void event2_set_children(event2 * e,event2 * children, int n_events)

def init():
    init_events()

def start():
    setup()

def stop():
    teardown()

def usecs():
    '''
    Get the current time in microseconds, according to JACK
    '''
    return get_time()

def frames():
    '''
    Get the current time in frames, according to JACK
    '''
    return get_frame_time()

def put(np.ndarray[FLOAT_DTYPE_t,ndim = 1] n,int starts, int stops,UINT64_DTYPE_t time):
    put_event(<float*>n.data,starts,stops,time,0)

def cancel_all():
    '''
    Cancel all pending events
    '''
    cancel_all_events()

# TODO: What about just playing a buffer, a la Z.play(frames.audio)?


cdef transform * build_transforms(e,int samplerate):
    '''
    Build C structures representing the transforms for a single event
    '''
    # loop counters
    cdef int i = 0,j = 0
    
    # the number of transforms defined for this event
    cdef int n_transforms = len(e.transforms)
    
    # allocate a block of contiguous memory for the transforms
    cdef transform * transforms = \
        <transform*>malloc(n_transforms * sizeof(transform))
    
    cdef parameter * parameters
    cdef int n_values
    cdef np.ndarray[FLOAT_DTYPE_t,ndim = 1] values
    cdef np.ndarray[UINT64_DTYPE_t,ndim = 1] times
    cdef np.ndarray[CHAR_DTYPE_t,ndim = 1] interpolations
    
    
    for i,t in enumerate(e.transforms):
        n_parameters = t.n_args
        # allocate a block of contiguous memory for the parameters
        parameters = <parameter*>malloc(n_parameters * sizeof(parameter))
        
        for j,p in enumerate(t.c_args):
            # initialize each parameter
            n_values,values,times,interpolations = p
            parameter_init(\
                &(parameters[j]),
                <float*>values.data,
                n_values,
                <UINT64_DTYPE_t*>times.data,
                <char*>interpolations.data)
        
        # initialize each transform
        # KLUDGE: There's gotta be a more elegant way to map Transform-derived
        # classes to C transform initializers
        # KLUDGE: What about transforms that take non-parameter constructor
        # args, e.g., filter type for a BiQuadFilter?  In other words, how do
        # I handle parameters that can't be automated and stay constant for the
        # lifetime of the transform?
        transform_name = t.__class__.__name__
        if 'Gain' == transform_name:
            gain_init(&(transforms[i]),parameters)
        elif 'Delay' == transform_name:
            # max delay time of 2 seconds at 44100 hz
            # KLUDGE: Another hard-coded sample rate
            delay_init(&(transforms[i]),samplerate * 2,parameters)
    
    return transforms
        
# KLUDGE: is the time parameter necessary? Relative times are now handled by
# the JACK client, I think. 
def enqueue(ptrn,buffers,int samplerate,patterns = None,time = None,parent_event = None):
    '''
    Add a pattern to the queue
    '''
    # Recursively build the C data structure, allocating buffers as we go, and
    # translating times into frames/samples which are relative to the parent.
    # When the data structure is complete, enqueue the top-level events
    
    cdef UINT64_DTYPE_t start_time_frames
    cdef float latency = 0.25 * samplerate
    cdef event2 * e
    cdef np.ndarray[FLOAT_DTYPE_t,ndim = 1] audio
    
    if ptrn.is_leaf and None is parent_event:
        # enqueue() was called directly on a leaf pattern
        # ensure that the audio has been fetched from the database
        audio = buffers.allocate(ptrn)
        # KLUDGE: What about other sample rates?
        # KLUDGE: Latency should be configurable
        start_time_frames = <UINT64_DTYPE_t>(get_frame_time() + latency)
        # The JACK client will free this memory once the event has played
        e = event2_new_buffer(<float*>audio.data,0,len(audio),start_time_frames)
        put_event2(e)  
    
    if None is patterns:
        patterns = ptrn.patterns
    
    # transform events from dict(pattern -> [events,....]) to
    # flat list of two-tuples of (pattern,event) 
    children = []
    for k,v in ptrn.pdata.iteritems():
        p = patterns[k]
        for evt in v:
            children.append((p,evt))
    
    
    cdef int i = 0
    cdef int n_children = len(children)
    cdef event2 * events = <event2*>malloc(n_children * sizeof(event2))
    cdef int n_transforms = 0
    cdef transform * transforms
    cdef float start_time_seconds
    
    for i,c in enumerate(children):
        p,evt = c
        
        n_transforms = len(evt.transforms)
        transforms = build_transforms(evt,samplerate)
        start_time_seconds = ptrn.interpret_time(evt.time) 
        start_time_frames = <UINT64_DTYPE_t>(start_time_seconds * samplerate)
        
        if p.is_leaf:
            audio = buffers.allocate(p)
            event2_new_leaf(&(events[i]),<float*>audio.data,0,len(audio),
                            start_time_frames,transforms,n_transforms)
        else:
            event2_new_base(\
                        &(events[i]),transforms,n_transforms,start_time_frames)
            enqueue(p,buffers,samplerate,patterns = patterns,
                    time = start_time_seconds,parent_event = events[i])
    
    cdef event2 * pe
    # TODO: if parent_event, attach events to parent
    if None is not parent_event:
        pe = <event2*>parent_event
        event2_set_children(pe,events,n_children)
            
    
    if None is parent_event:
        # get the current time and schedule the child events of this pattern
        # schedule the child events of this pattern
        now = get_frame_time()
        for i in range(n_children):
            events[i].start_time_frames += <UINT64_DTYPE_t>(now + latency)
            put_event2(&(events[i]))
    
    
    
    