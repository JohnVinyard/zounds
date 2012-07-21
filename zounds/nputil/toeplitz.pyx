from __future__ import division
cimport numpy as np

INT_DTYPE = np.int
ctypedef np.int_t INT_DTYPE_t

FLOAT_DTYPE = np.float32
ctypedef np.float32_t FLOAT_DTYPE_t

cimport cython
@cython.boundscheck(False)
def toeplitz2dc(np.ndarray[FLOAT_DTYPE_t,ndim = 2] patch,size):
    '''
    Construct a matrix that makes convolution possible via a matrix-vector
    operation, similar to constructing a toeplitz matrix for 1d signals
    '''
    # width of the patch
    cdef int pw = patch.shape[0]
    # total size of the patch
    cdef int patchsize = patch.size
    # width of the kernel
    cdef int kw = size[0]
    # height of the kernel
    cdef int kh = size[1]
    # size of the kernel
    cdef int ksize = np.product(size)
    
    # the patch width, without boundaries
    cdef int w = patch.shape[0] - kw
    # the patch height, without boundaries
    cdef int h = patch.shape[1] - kh
    # the total number of positions to visit
    cdef int totalsize = w * h
    # this will hold each section of the patch
    cdef np.ndarray[FLOAT_DTYPE_t,ndim = 2] l = \
        np.zeros((totalsize,ksize),dtype = FLOAT_DTYPE)
    cdef int c = 0
    cdef int j = 0
    cdef int i = 0
    cdef int s = 0
    
    cdef int x = 0
    cdef int y = 0
    cdef int z = 0
    
    for s in range(patchsize):
        j = int(s / pw)
        i = s % pw
        if i < w and j < h:
            z = 0
            for x in range(i,i + kw):
                for y in range(j,j + kh):
                    l[c,z] = patch[x,y]
                    z += 1
            c += 1
    return l