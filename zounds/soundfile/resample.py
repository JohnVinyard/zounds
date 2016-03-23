from __future__ import division

from ctypes import *

import numpy as np

from zounds.timeseries import AudioSamples, SR44100

libsamplerate = CDLL('libsamplerate.so')

from featureflow import Node


class SRC_DATA(Structure):
    '''
    A wrapper for the libsamplerate.SRC_DATA struct
    '''
    _fields_ = [('data_in', POINTER(c_float)),
                ('data_out', POINTER(c_float)),
                ('input_frames', c_long),
                ('output_frames', c_long),
                ('input_frames_used', c_long),
                ('output_frames_gen', c_long),
                ('end_of_input', c_int),
                ('src_ratio', c_double), ]


class Resample(object):
    """
    A wrapper around the libsamplerate src_process() method.  This class is
    intended for one-time use. New instances should be created for each sound\
    file processed.
    """

    def __init__(self, orig_sample_rate, new_sample_rate, \
                 nchannels=1, converter_type=1):
        """
        orig_sample_rate - The sample rate of the incoming samples, in hz
        new_sample_rate - The sample_rate of the outgoiing samples, in hz
        n_channels - Number of channels in the incoming and outgoing samples
        converter_type - See http://www.mega-nerd.com/SRC/api_misc.html#Converters
                         for a list of conversion types. "0" is the best-quality,
                         and slowest converter

        """
        self._ratio = new_sample_rate / orig_sample_rate
        # check if the conversion ratio is considered valid by libsamplerate
        if not libsamplerate.src_is_valid_ratio(c_double(self._ratio)):
            raise ValueError('%1.2f / %1.2f = %1.4f is not a valid ratio' % \
                             (new_sample_rate, orig_sample_rate, self._ratio))
        # create a pointer to the SRC_STATE struct, which maintains state
        # between calls to src_process()
        error = pointer(c_int(0))
        self.nchannels = nchannels
        self.converter_type = converter_type
        self._state = libsamplerate.src_new( \
                c_int(converter_type), c_int(nchannels), error)

    def _prepare_input(self, insamples):
        # ensure that the input is float data
        if np.float32 != insamples.dtype:
            insamples = insamples.astype(np.float32)

        return insamples

    def _output_buffer(self, insamples):
        outsize = int(np.round(insamples.size * self._ratio))
        return np.zeros(outsize, dtype=np.float32)

    def _src_data_struct(self, insamples, outsamples, end_of_input=False):
        # Build the SRC_DATA struct
        return SRC_DATA( \
                # a pointer to the input samples
                data_in=insamples.ctypes.data_as(POINTER(c_float)),
                # a pointer to the output buffer
                data_out=outsamples.ctypes.data_as(POINTER(c_float)),
                # number of input samples
                input_frames=insamples.size,
                # number of output samples
                output_frames=outsamples.size,
                # NOT the end of input, i.e., there is more data to process
                end_of_input=int(end_of_input),
                # the conversion ratio
                src_ratio=self._ratio)

    def _check_for_error(self, return_code):
        if return_code:
            raise Exception( \
                    c_char_p(
                        libsamplerate.src_strerror(c_int(return_code))).value)

    def all_at_once(self, insamples):
        insamples = self._prepare_input(insamples)
        outsamples = self._output_buffer(insamples)
        sd = self._src_data_struct(insamples, outsamples)
        rv = libsamplerate.src_simple(pointer(sd),
                                      c_int(self.converter_type),
                                      c_int(self.nchannels))
        self._check_for_error(rv)
        return outsamples

    def __call__(self, insamples, end_of_input=False):
        insamples = self._prepare_input(insamples)
        outsamples = self._output_buffer(insamples)
        sd = self._src_data_struct(insamples, outsamples, end_of_input)
        rv = libsamplerate.src_process(self._state, pointer(sd))
        self._check_for_error(rv)
        return outsamples


class Resampler(Node):
    def __init__(self, samplerate=None, needs=None):
        super(Resampler, self).__init__(needs=needs)
        self._samplerate = samplerate or SR44100()
        self._resample = None

    def _noop(self, data, finalized):
        return data

    def _process(self, data):
        sr = data.samples_per_second

        if self._resample is None:
            target_sr = self._samplerate.samples_per_second
            self._resample = Resample( \
                    sr,
                    target_sr,
                    1 if len(data.shape) == 1 else data.shape[1])

            if target_sr != sr:
                self._rs = self._resample
                # KLUDGE: The following line seems to solve a bug whereby 
                # libsamplerate doesn't generate enough samples the first time
                # src_process is called. We're calling it once here, so the "real"
                # output will come out click-free
                self._resample(np.zeros(target_sr, dtype=np.float32))
            else:
                self._rs = self._noop

        resampled = self._rs(data, self._finalized)
        if not isinstance(resampled, AudioSamples):
            resampled = AudioSamples( \
                    resampled, self._samplerate)
        yield resampled
