from __future__ import division
cimport numpy as np
import numpy as np

INT_DTYPE = np.int
ctypedef np.int_t INT_DTYPE_t

FLOAT_DTYPE = np.float32
ctypedef np.float32_t FLOAT_DTYPE_t

UINT64_DTYPE = np.uint64
ctypedef np.uint64_t UINT64_DTYPE_t

ctypedef unsigned long ULong

cimport cython

@cython.boundscheck(False)
def count_bits(np.ndarray[UINT64_DTYPE_t,ndim = 1] n):
    cdef np.ndarray[INT_DTYPE_t,ndim = 1] out = \
        np.ndarray(n.size,dtype = INT_DTYPE)
    cdef int i = 0
    cdef int l = n.size
    cdef UINT64_DTYPE_t a = 0x5555555555555555
    cdef UINT64_DTYPE_t b = 0x3333333333333333
    cdef UINT64_DTYPE_t c = 0xF0F0F0F0F0F0F0F
    cdef UINT64_DTYPE_t d = 0x101010101010101
    cdef UINT64_DTYPE_t z = 0
    for i in range(l):
        z = n[i]
        z = z - ((z >> 1) & a)
        z = (z & b) + ((z >> 2) & b)
        out[i] = ((z + (z >> 4)) & c) * d >> 56
    return out


def count_packed_bits(np.ndarray[UINT64_DTYPE_t,ndim = 2] n):
    cdef int ns = n.shape[0]
    cdef int ns2 = n.shape[1]
    cdef np.ndarray[INT_DTYPE_t,ndim = 1] out = \
        np.ndarray(ns,dtype = INT_DTYPE)
    cdef int i = 0
    cdef int j = 0
    cdef int q = 0
    cdef UINT64_DTYPE_t a = 0x5555555555555555
    cdef UINT64_DTYPE_t b = 0x3333333333333333
    cdef UINT64_DTYPE_t c = 0xF0F0F0F0F0F0F0F
    cdef UINT64_DTYPE_t d = 0x101010101010101
    cdef UINT64_DTYPE_t z = 0
    for i in range(ns):
        z = 0
        for j in range(ns2):
            q = n[i][j]
            q = q - ((q >> 1) & a)
            q = (q & b) + ((q >> 2) & b)
            z += ((q + (q >> 4)) & c) * d >> 56
        out[i] = z
    return out
    

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