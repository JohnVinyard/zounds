
cdef inline int fft_index(float freq_hz,int ws,int sr):
    cdef float fft_bandwidth
    fft_bandwidth = (sr * .5) / (ws * .5)
    return 