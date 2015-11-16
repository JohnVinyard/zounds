from flow import Node, Feature, Decoder, NotEnoughData
from timeseries import ConstantRateTimeSeries, TimeSlice
import numpy as np
from duration import Picoseconds
import struct

class MeasureOfTransience(Node):
    '''
    Measure of Transience, as defined in section 5.2.1 of 
    http://www.mp3-tech.org/programmer/docs/Masri_thesis.pdf
    '''
    def __init__(self, needs = None):
        super(MeasureOfTransience, self).__init__(needs = needs)
    
    def _first_chunk(self, data):
        self._bin_numbers = np.arange(1, data.shape[1] + 1)
        padding = np.zeros(data.shape[1])
        padding[:] = 1e-12
        data = np.concatenate([padding[None,:], data])
    
    def _process(self, data):
        magnitude = (data[:, 2:] ** 2)
        energy = magnitude.sum(axis = 1)
        hfc = (magnitude * self._bin_numbers[2:]).sum(axis = 1)
        energy[energy == 0] = 1e-12
        hfc[hfc == 0] = 1e-12
        mot = (hfc[1:] / hfc[:-1]) * (hfc[1:] / energy[1:])
        yield ConstantRateTimeSeries(mot, data.frequency, data.duration)

class ComplexDomain(Node):
    '''
    Complex-domain onset detection as described in
    http://www.eecs.qmul.ac.uk/legacy/dafx03/proceedings/pdfs/dafx81.pdf
    '''
    def __init__(self, needs = None):
        super(ComplexDomain, self).__init__(needs = needs)
    
    def _first_chunk(self, data):
        first = np.zeros((2, 3, data.shape[-1]), dtype = np.complex128)
        first[0, 2:, :] = data[0, :1]
        first[1, 1:, :] = data[0, :2]
        return ConstantRateTimeSeries(\
              np.concatenate([first, data]),
              data.frequency,
              data.duration)
    
    def _process(self, data):
        # delta between expected and actual phase
        # TODO: unwrap phases before computing deltas, to avoid artifacts
        # or discontinuties from phase boundary wrapping
        angle = np.angle(data[:,2] - (2 * data[:,1]) + data[:,0])
        
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
          ((expected**2) + (actual**2) - 
          (2 * expected * actual * np.cos(angle))) ** 0.5)[nonzero_phase_delta_indices]
          
        # TODO: This duration isn't right.  It should probably be 
        # data.duration // 3
        yield ConstantRateTimeSeries(\
             detect.sum(axis = 1),
             data.frequency,
             data.duration)
        
class Flux(Node):
    
    def __init__(self, needs = None):
        super(Flux, self).__init__(needs = needs)
        self._memory = None
    
    def _process(self, data):
        
        if self._memory is None:
            # prepend the first vector, so that the initial flux value is zero
            d = np.vstack([data[0], data])
        else:
            # prepend the last vector from the previous batch
            d = np.vstack([self._memory, data])
        
        self._memory = data[-1]
        
        # Take the difference, keeping only positive changes 
        # (the magnitude increased)
        diff = np.diff(d, axis = 0)
        diff[diff < 0] = 0
        
        # take the l1 norm of each frame
        yield ConstantRateTimeSeries(\
              diff.sum(axis = 1),
              data.frequency,
              data.duration)

class BasePeakPicker(Node):
    
    def __init__(self, needs = None):
        super(BasePeakPicker, self).__init__(needs = needs)
        self._pos = Picoseconds(0)
    
    def _onset_indices(self, data):
        raise NotImplementedError()
    
    def _process(self, data):
        indices = self._onset_indices(data)
        timestamps = self._pos + (indices * data.frequency)
        self._pos += len(data) * data.frequency
        yield timestamps
        
        if self._finalized:
            yield self._pos

class MovingAveragePeakPicker(Node):
    
    def __init__(self, aggregate = np.mean, needs = None):
        super(MovingAveragePeakPicker, self).__init__(needs = needs)
        self._aggregate = aggregate
    
    def _first_chunk(self, data):
        self._center = data.shape[1] // 2
    
    def _onset_indices(self, data):
        agg = self._aggregate(data, axis = 1)
        return np.where(data[:, self._center] > agg)[0]

class PeakPicker(BasePeakPicker):
    
    def __init__(self, factor = 3.5, needs = None):
        super(PeakPicker, self).__init__(needs = needs)
        self._factor = factor
    
    def _onset_indices(self, data):
        mean = data.mean(axis = 1) * self._factor
        return np.where(data[:,0] > mean)[0]

class SparseTimestampEncoder(Node):
    
    content_type = 'application/octet-stream'
    
    def __init__(self, needs = None):
        super(SparseTimestampEncoder, self).__init__(needs = needs)
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
        data = np.fromstring(flo.read(), dtype = np.uint64)
        return np.array(data, dtype = dtype)
    
    def __iter__(self, flo):
        yield self(flo)

class TimeSliceDecoder(SparseTimestampDecoder):
    
    def __init__(self):
        super(TimeSliceDecoder, self).__init__()
    
    def __call__(self, flo):
        timestamps = super(TimeSliceDecoder, self).__call__(flo)
        durations = np.diff(timestamps)
        return (TimeSlice(d, s) for s, d in zip(timestamps, durations))
        
    def __iter__(self, flo):
        yield self(flo)


class SparseTimestampFeature(Feature):
    
    def __init__(\
        self,
        extractor,
        needs = None,
        store = False,
        key = None,
        encoder = SparseTimestampEncoder,
        decoder = SparseTimestampDecoder(),
        **extractor_args):
        
        super(SparseTimestampFeature ,self).__init__(\
            extractor,
            needs = needs,
            store = store,
            encoder = encoder,
            decoder = decoder,
            key = key,
            **extractor_args)

class TimeSliceFeature(Feature):
    
    def __init__(\
        self,
        extractor,
        needs = None,
        store = False,
        key = None,
        encoder = SparseTimestampEncoder,
        decoder = TimeSliceDecoder(),
        **extractor_args):
        
        super(TimeSliceFeature ,self).__init__(\
            extractor,
            needs = needs,
            store = store,
            encoder = encoder,
            decoder = decoder,
            key = key,
            **extractor_args)
    

        