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


# TODO: Why doesn't this function have boundscheck = False too?
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