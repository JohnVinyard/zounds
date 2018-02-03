from __future__ import division

from ctypes import *

import numpy as np

from zounds.timeseries import SR44100, AudioSamples, Seconds
from zounds.core import ArrayWithUnits

try:
    libsamplerate = CDLL('libsamplerate.so')
except OSError, e:
    # KLUDGE: This is here to support building documentation on readthedocs
    pass

from featureflow import Node


class SRC_DATA(Structure):
    """
    A wrapper for the libsamplerate.SRC_DATA struct
    """
    _fields_ = [('data_in', POINTER(c_float)),
                ('data_out', POINTER(c_float)),
                ('input_frames', c_long),
                ('output_frames', c_long),
                ('input_frames_used', c_long),
                ('output_frames_gen', c_long),
                ('end_of_input', c_int),
                ('src_ratio', c_double), ]


class SRC_STATE(Structure):
    """
    A dummy structure to represent the state returned from libsamplerate
    src_new.
    """
    _fields_ = []


class Resample(object):
    """
    A wrapper around the libsamplerate src_process() method.  This class is
    intended for one-time use. New instances should be created for each sound\
    file processed.
    """

    def __init__(
            self,
            orig_sample_rate,
            new_sample_rate,
            nchannels=1,
            converter_type=1):

        """
        orig_sample_rate - The sample rate of the incoming samples, in hz
        new_sample_rate - The sample_rate of the outgoiing samples, in hz
        n_channels - Number of channels in the incoming and outgoing samples
        converter_type - See http://www.mega-nerd.com/SRC/api_misc.html#Converters
                         for a list of conversion types. "0" is the best-quality,
                         and slowest converter

        """
        super(Resample, self).__init__()
        self._ratio = new_sample_rate / orig_sample_rate
        print self._ratio
        # check if the conversion ratio is considered valid by libsamplerate
        if not libsamplerate.src_is_valid_ratio(c_double(self._ratio)):
            raise ValueError('%1.2f / %1.2f = %1.4f is not a valid ratio' % \
                             (new_sample_rate, orig_sample_rate, self._ratio))
        # create a pointer to the SRC_STATE struct, which maintains state
        # between calls to src_process()
        self.error = pointer(c_int(0))
        self.nchannels = nchannels
        self.converter_type = converter_type
        self.c_int_converter_type = c_int(converter_type)
        self.c_int_channels = c_int(self.nchannels)
        libsamplerate.src_new.restype = POINTER(SRC_STATE)
        self._state = libsamplerate.src_new(
            self.c_int_converter_type, self.c_int_channels, self.error)

    def _prepare_input(self, insamples):
        # ensure that the input is float data
        if np.float32 != insamples.dtype:
            return insamples.astype(np.float32)
        return insamples

    def _output_buffer(self, insamples):
        outsize = (int(np.round(len(insamples) * self._ratio)), self.nchannels)
        return np.zeros(outsize, dtype=np.float32).squeeze()

    def _check_for_error(self, return_code):
        if return_code:
            raise Exception(
                'libsamplerate sent non-zero return code {return_code}'
                    .format(**locals()))

    def __call__(self, insamples, end_of_input=False):

        normalized_insamples = self._prepare_input(insamples)
        outsamples = self._output_buffer(normalized_insamples)

        insamples_ptr = normalized_insamples.ctypes.data_as(POINTER(c_float))
        outsamples_ptr = outsamples.ctypes.data_as(POINTER(c_float))

        sd = SRC_DATA(
            # a pointer to the input samples
            data_in=insamples_ptr,
            # a pointer to the output buffer
            data_out=outsamples_ptr,
            # number of input samples
            input_frames=len(normalized_insamples),
            # number of output samples
            output_frames=len(outsamples),
            # NOT the end of input, i.e., there is more data to process
            end_of_input=int(end_of_input),
            # the conversion ratio
            src_ratio=self._ratio)
        sd_ptr = pointer(sd)
        rv = libsamplerate.src_process(self._state, sd_ptr)
        self._check_for_error(rv)
        return outsamples


class Resampler(Node):
    """
    `Resampler` expects to process :class:`~zounds.timeseries.AudioSamples`
    instances (e.g., those produced by a :class:`AudioStream` node), and will
    produce a new stream of :class:`AudioSamples` at a new sampling rate.

    Args:
        samplerate (AudioSampleRate): the desired sampling rate.  If none is
            provided, the default is :class:`~zounds.timeseries.SR44100`
        needs (Feature): a processing node that produces
            :class:`~zounds.timeseries.AudioSamples`


    Here's how you'd typically see :class:`Resampler` used in a processing
    graph.

    .. code:: python

        import featureflow as ff
        import zounds

        chunksize = zounds.ChunkSizeBytes(
            samplerate=zounds.SR44100(),
            duration=zounds.Seconds(30),
            bit_depth=16,
            channels=2)

        @zounds.simple_in_memory_settings
        class Document(ff.BaseModel):
            meta = ff.JSONFeature(
                zounds.MetaData,
                store=True,
                encoder=zounds.AudioMetaDataEncoder)

            raw = ff.ByteStreamFeature(
                ff.ByteStream,
                chunksize=chunksize,
                needs=meta,
                store=False)

            pcm = zounds.AudioSamplesFeature(
                zounds.AudioStream,
                needs=raw,
                store=True)

            resampled = zounds.AudioSamplesFeature(
                zounds.Resampler,
                samplerate=zounds.SR22050(),
                needs=pcm,
                store=True)


        synth = zounds.NoiseSynthesizer(zounds.SR11025())
        samples = synth.synthesize(zounds.Seconds(10))
        raw_bytes = samples.encode()
        _id = Document.process(meta=raw_bytes)
        doc = Document(_id)
        print doc.pcm.samplerate.__class__.__name__  # SR11025
        print doc.resampled.samplerate.__class__.__name__  # SR22050
    """

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
            self._resample = Resample(
                sr,
                target_sr,
                1 if len(data.shape) == 1 else data.shape[1])

            if target_sr != sr:
                self._rs = self._resample
                # KLUDGE: The following line seems to solve a bug whereby 
                # libsamplerate doesn't generate enough samples the first time
                # src_process is called. We're calling it once here, so the "real"
                # output will come out click-free
                silence = AudioSamples.silence(
                    self._samplerate, Seconds(1), channels=data.channels)
                self._resample(silence)
            else:
                self._rs = self._noop

        resampled = self._rs(data, self._finalized)
        if not isinstance(resampled, ArrayWithUnits):
            resampled = AudioSamples(resampled, self._samplerate)
        yield resampled
