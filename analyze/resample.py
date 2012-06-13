from __future__ import division
from ctypes import *
import numpy as np
libsamplerate = CDLL('libsamplerate.so')

class SRC_DATA(Structure):
    '''
    A wrapper for the libsamplerate.SRC_DATA struct
    '''
    _fields_ = [('data_in',POINTER(c_float)),
                ('data_out',POINTER(c_float)),
                ('input_frames',c_long),
                ('output_frames',c_long),
                ('input_frames_used',c_long),
                ('output_frames_gen',c_long),
                ('end_of_input',c_int),
                ('src_ratio',c_double),]


    
class Resample(object):
    '''
    A wrapper around the libsamplerate src_process() method.  This class is 
    intended for one-time use. New instances should be created for each sound\
    file processed.
    '''    
    def __init__(self,orig_sample_rate,new_sample_rate,\
                 nchannels = 1, converter_type = 0):
        '''
        orig_sample_rate - The sample rate of the incoming samples, in hz
        new_sample_rate - The sample_rate of the outgoiing samples, in hz
        n_channels - Number of channels in the incoming and outgoing samples
        converter_type - See http://www.mega-nerd.com/SRC/api_misc.html#Converters 
                         for a list of conversion types. "0" is the best-quality,
                         and slowest converter
        
        '''
        print 'resampling from %i to %i' % (orig_sample_rate,new_sample_rate)
        self._ratio = new_sample_rate / orig_sample_rate
        # check if the conversion ratio is considered valid by libsamplerate
        if not libsamplerate.src_is_valid_ratio(c_double(self._ratio)):
            raise ValueError('%1.2f / %1.2f = %1.4f is not a valid ratio' % \
                             (new_sample_rate,orig_sample_rate,self._ratio))
        # create a pointer to the SRC_STATE struct, which maintains state
        # between calls to src_process()
        error = pointer(c_int(0))
        self._state = libsamplerate.src_new(\
                                c_int(converter_type),c_int(nchannels),error)
    
    def __call__(self,insamples,end_of_input = False):
        # ensure that the input is float data
        if np.float32 != insamples.dtype:
            insamples = insamples.astype(np.float32)
        
        outsize = int(np.round(insamples.size * self._ratio))
        outsamples = np.zeros(outsize,dtype = np.float32)
        # Build the SRC_DATA struct
        sd = SRC_DATA(\
                # a pointer to the input samples
                data_in = insamples.ctypes.data_as(POINTER(c_float)),
                # a pointer to the output buffer
                data_out = outsamples.ctypes.data_as(POINTER(c_float)),
                # number of input samples
                input_frames = insamples.size,
                # number of output samples
                output_frames = outsize,
                # NOT the end of input, i.e., there is more data to process
                end_of_input = int(end_of_input),
                # the conversion ratio
                src_ratio = self._ratio)
        
        # Check for a non-zero return code after each src_process() call
        rv = libsamplerate.src_process(self._state,pointer(sd))
        if rv:
            # print the string error for the non-zero return code
            raise Exception(c_char_p(libsamplerate.src_strerror(c_int(rv))).value)
        
        
        return outsamples