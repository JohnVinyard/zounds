import numpy as np
from featureflow import Node, Feature

from zounds.nputil import safe_unit_norm
from zounds.timeseries import \
    TimeSlice, Picoseconds, TimeDimension, VariableRateTimeSeries, \
    VariableRateTimeSeriesEncoder, VariableRateTimeSeriesDecoder
from zounds.core import ArrayWithUnits


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

        return ArrayWithUnits(
            np.concatenate([padding[None, :], data]), data.dimensions)

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

        self._cache = ArrayWithUnits(
            self._cache[None, -1], self._cache.dimensions)

        return data

    def _process(self, data):
        data = np.abs(data)
        magnitude = (data[:, 2:] ** 2)
        energy = magnitude.sum(axis=1)
        hfc = (magnitude * self._bin_numbers[2:]).sum(axis=1)
        energy[energy == 0] = 1e-12
        hfc[hfc == 0] = 1e-12
        mot = (hfc[1:] / hfc[:-1]) * (hfc[1:] / energy[1:])
        yield mot


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

        dims = \
            [TimeDimension(data.frequency, data.duration // 3)] \
            + data.dimensions[1:]
        output = ArrayWithUnits(detect.sum(axis=1), dims)
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

        self._cache = ArrayWithUnits(self._cache, data.dimensions)

        self._memory = data[-1]

    def _process(self, data):
        if self._unit_norm:
            data = safe_unit_norm(data)
        diff = np.diff(data, axis=0)

        yield ArrayWithUnits(
            np.linalg.norm(diff, axis=-1), data.dimensions)


class BasePeakPicker(Node):
    def __init__(self, needs=None):
        super(BasePeakPicker, self).__init__(needs=needs)
        self._pos = Picoseconds(0)
        self._leftover_timestamp = self._pos

    def _onset_indices(self, data):
        raise NotImplementedError()

    def _last_chunk(self):
        yield VariableRateTimeSeries((
            (TimeSlice(
             start=self._leftover_timestamp,
             duration=self._pos - self._leftover_timestamp), np.zeros(0)),
        ))

    def _process(self, data):
        td = data.dimensions[0]
        frequency = td.frequency

        indices = self._onset_indices(data)
        timestamps = self._pos + (indices * frequency)
        self._pos += len(data) * frequency

        timestamps = [self._leftover_timestamp] + list(timestamps)
        self._leftover_timestamp = timestamps[-1]

        time_slices = TimeSlice.slices(timestamps)
        vrts = VariableRateTimeSeries([(ts, np.zeros(0)) for ts in time_slices])
        yield vrts


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


class TimeSliceFeature(Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=VariableRateTimeSeriesEncoder,
            decoder=VariableRateTimeSeriesDecoder(),
            **extractor_args):
        super(TimeSliceFeature, self).__init__(
            extractor,
            needs=needs,
            store=store,
            encoder=encoder,
            decoder=decoder,
            key=key,
            **extractor_args)
