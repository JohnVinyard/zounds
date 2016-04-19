import struct

import numpy as np
from featureflow import Node, Feature, Decoder

from zounds.nputil import safe_unit_norm
from zounds.timeseries import ConstantRateTimeSeries, TimeSlice, Picoseconds


class MeasureOfTransience(Node):
    """
    Measure of Transience, as defined in section 5.2.1 of
    http://www.mp3-tech.org/programmer/docs/Masri_thesis.pdf

    Uses the ratio of high-frequency content in the signal to detect onsets.
    Effective for percussive onsets.
    """

    def __init__(self, needs=None):
        super(MeasureOfTransience, self).__init__(needs=needs)

    def _first_chunk(self, data):
        data = np.abs(data)
        self._bin_numbers = np.arange(1, data.shape[1] + 1)
        padding = np.zeros(data.shape[1])
        padding[:] = data[0]
        return ConstantRateTimeSeries(
                np.concatenate([padding[None, :], data]),
                data.frequency,
                data.duration)

    # TODO: this pattern of hanging on to the last sample of the previous chunk,
    # and appending to the next chunk probably happens somewhere else, and
    # should be generalized
    def _enqueue(self, data, pusher):
        if self._cache is None:
            self._cache = data
        else:
            self._cache = self._cache.concatenate(data)

    def _dequeue(self):
        data = self._cache
        self._cache = self._cache[None, -1]
        return data

    def _process(self, data):
        data = np.abs(data)
        magnitude = (data[:, 2:] ** 2)
        energy = magnitude.sum(axis=1)
        hfc = (magnitude * self._bin_numbers[2:]).sum(axis=1)
        energy[energy == 0] = 1e-12
        hfc[hfc == 0] = 1e-12
        mot = (hfc[1:] / hfc[:-1]) * (hfc[1:] / energy[1:])
        yield ConstantRateTimeSeries(mot, data.frequency, data.duration)


class ComplexDomain(Node):
    """
    Complex-domain onset detection as described in
    http://www.eecs.qmul.ac.uk/legacy/dafx03/proceedings/pdfs/dafx81.pdf
    """

    def __init__(self, needs=None):
        super(ComplexDomain, self).__init__(needs=needs)

    def _process(self, data):
        # delta between expected and actual phase
        # TODO: unwrap phases before computing deltas, to avoid artifacts
        # or discontinuties from phase boundary wrapping
        angle = np.angle(data)
        angle = np.unwrap(angle, axis=1)
        angle = np.angle(angle[:, 2] - (2 * angle[:, 1]) + angle[:, 0])

        # expected magnitude
        expected = np.abs(data[:, 1, :])
        # actual magnitude
        actual = np.abs(data[:, 2, :])
        # detection function array
        detect = np.zeros(angle.shape)

        # where phase delta is zero, detection function is the difference 
        # between expected and actual magnitude
        zero_phase_delta_indices = np.where(angle == 0)
        detect[zero_phase_delta_indices] = \
            (expected - actual)[zero_phase_delta_indices]

        # where phase delta is non-zero, detection function combines magnitude
        # and phase deltas
        nonzero_phase_delta_indices = np.where(angle != 0)
        detect[nonzero_phase_delta_indices] = (
            ((expected ** 2) + (actual ** 2) -
             (2 * expected * actual * np.cos(angle))) ** 0.5)[
            nonzero_phase_delta_indices]

        output = ConstantRateTimeSeries(
                detect.sum(axis=1),
                data.frequency,
                data.duration // 3)
        yield output


class Flux(Node):
    def __init__(self, unit_norm=False, needs=None):
        super(Flux, self).__init__(needs=needs)
        self._memory = None
        self._unit_norm = unit_norm

    def _enqueue(self, data, pusher):
        if self._memory is None:
            self._cache = np.vstack((data[0], data))
        else:
            self._cache = np.vstack((self._memory, data))
        self._cache = ConstantRateTimeSeries( \
                self._cache,
                data.frequency,
                data.duration)
        self._memory = data[-1]

    def _process(self, data):
        if self._unit_norm:
            data = safe_unit_norm(data)
        diff = np.diff(data, axis=0)
        yield ConstantRateTimeSeries( \
                np.linalg.norm(diff, axis=-1),
                data.frequency,
                data.duration)


class BasePeakPicker(Node):
    def __init__(self, needs=None):
        super(BasePeakPicker, self).__init__(needs=needs)
        self._pos = Picoseconds(0)

    def _onset_indices(self, data):
        raise NotImplementedError()

    def _last_chunk(self):
        yield self._pos

    def _process(self, data):
        if self._pos == Picoseconds(0):
            yield self._pos

        indices = self._onset_indices(data)
        timestamps = self._pos + (indices * data.frequency)
        self._pos += len(data) * data.frequency
        yield timestamps


class MovingAveragePeakPicker(BasePeakPicker):
    def __init__(self, aggregate=np.mean, needs=None):
        super(MovingAveragePeakPicker, self).__init__(needs=needs)
        self._aggregate = aggregate

    def _first_chunk(self, data):
        self._center = data.shape[1] // 2
        return data

    def _onset_indices(self, data):
        # compute the threshold for onsets
        agg = self._aggregate(data, axis=1) * 1.25
        # find indices that are peaks
        diff = np.diff(data[:, self._center - 2: self._center + 1])
        peaks = (diff[:, 0] > 0) & (diff[:, 1] < 0)
        # find indices that are above the local average or median
        over_thresh = data[:, self._center] > agg
        # return the intersection of the two
        return np.where(peaks & over_thresh)[0]


class SparseTimestampEncoder(Node):
    content_type = 'application/octet-stream'

    def __init__(self, needs=None):
        super(SparseTimestampEncoder, self).__init__(needs=needs)
        # TODO: Add a class (mixin) in the flow library for this pattern where
        # the _process implementarion changes depending on whether it's the first
        # call or a subsequent one
        self._initialized = False

    def _process(self, data):
        if not self._initialized:
            sd = str(data.dtype)
            yield struct.pack('B', len(sd))
            yield sd
            self._initialized = True

        yield data.astype(np.uint64).tostring()


# TODO: Encode/decode tests
# TODO: A subclass of this that turns each pair into a timeslice
# TODO: Should PeakPicker always emit the *end* of the timeseries, so that the
# final timeslice can be produced correctly?
class SparseTimestampDecoder(Decoder):
    def __init__(self):
        super(SparseTimestampDecoder, self).__init__()

    def __call__(self, flo):
        dtype_len = struct.unpack('B', flo.read(1))[0]
        dtype = np.dtype(flo.read(dtype_len))
        data = np.fromstring(flo.read(), dtype=np.uint64)
        return np.array(data, dtype=dtype)

    def __iter__(self, flo):
        yield self(flo)


class TimeSliceDecoder(SparseTimestampDecoder):
    def __init__(self):
        super(TimeSliceDecoder, self).__init__()

    def __call__(self, flo):
        timestamps = super(TimeSliceDecoder, self).__call__(flo)
        durations = np.diff(timestamps)
        return [TimeSlice(d, s) for s, d in zip(timestamps, durations)]

    def __iter__(self, flo):
        yield self(flo)


class SparseTimestampFeature(Feature):
    def __init__( \
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=SparseTimestampEncoder,
            decoder=SparseTimestampDecoder(),
            **extractor_args):
        super(SparseTimestampFeature, self).__init__( \
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)


class TimeSliceFeature(Feature):
    def __init__( \
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=SparseTimestampEncoder,
            decoder=TimeSliceDecoder(),
            **extractor_args):
        super(TimeSliceFeature, self).__init__( \
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
