from flow import Node, Feature, Decoder, NotEnoughData
from timeseries import ConstantRateTimeSeries, TimeSlice
import numpy as np
from duration import Picoseconds, Seconds
import struct

class Energy(Node):
    
    def __init__(self, needs = None):
        super(Energy, self).__init__(needs = needs)
    
    def _process(self, data):
        yield (data[:, 2:] ** 2).sum(axis = 1)

class HighFrequencyContent(Node):
    
    def __init__(self, needs = None):
        super(HighFrequencyContent, self).__init__(needs = needs)
        self._bin_numbers = None
    
    def _process(self, data):
        if self._bin_numbers is None:
            self._bin_numbers = np.arange(1, data.shape[1] + 1)
        yield ((data[:, 2:] ** 2) * self._bin_numbers[2:]).sum(axis = 1)

class MesaureOfTransience(Node):
    
    def __init__(self, needs = None):
        super(MesaureOfTransience, self).__init__(needs = needs)
    
    def _process(self, data):
        pass
        
        
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

class PeakPicker(Node):
    
    def __init__(self, factor = 3.5, needs = None):
        super(PeakPicker, self).__init__(needs = needs)
        self._factor = factor
        self._pos = Picoseconds(0)
    
    def _process(self, data):
        mean = data.mean(axis = 1) * self._factor
        indices = np.where(data[:,0] > mean)[0]
        timestamps = self._pos + (indices * data.frequency)
        self._pos += len(data) * data.frequency
        yield timestamps
        
        if self._finalized:
            yield self._pos

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
    

        