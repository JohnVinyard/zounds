from flow import NumpyMetaData, Node, Feature, Decoder
import numpy as np
from duration import Picoseconds 
from samplerate import SampleRate

class TimeSlice(object):
    
    def __init__(self, duration, start = None):
        super(TimeSlice, self).__init__()
        
        if not isinstance(duration, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = duration.__class__))
        
        if start != None and not isinstance(start, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = start.__class__))
        
        self.duration = duration
        self.start = start or np.timedelta64(0,'s')
    
    def __add__(self, other):
        return TimeSlice(self.duration, start = self.start + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    @property
    def end(self):
        return self.start + self.duration
    
    def __and__(self, other):
        delta = max(\
           np.timedelta64(0,'s'),
           min(self.end, other.end) - max(self.start, other.start))
        return TimeSlice(delta)

    def __contains__(self, other):
        if isinstance(other, np.timedelta64):
            return other > self.start and other < self.end
        if isinstance(other, TimeSlice):
            return other.start > self.start and other.end < self.end
        raise ValueError
    
    def __eq__(self, other):
        return self.start == other.start and self.duration == other.duration
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '{cls}(duration = {duration}, start = {start})'.format(\
              cls = self.__class__.__name__,
              duration = str(self.duration),
              start = str(self.start))
    
    def __str__(self):
        return self.__repr__()

class TimeSeries(object):
    '''
    TimeSeries interface
    '''
    def __init__(self):
        super(TimeSeries, self).__init__()

class ConstantRateTimeSeriesMetadata(NumpyMetaData):
    
    def __init__(\
         self, 
         dtype = None, 
         shape = None, 
         frequency = None, 
         duration = None):
        
        super(ConstantRateTimeSeriesMetadata, self).__init__(\
            dtype = dtype, shape = shape)
        self.frequency = self._decode_timedelta(frequency)
        self.duration = self._decode_timedelta(duration)
    
    @staticmethod
    def from_timeseries(timeseries):
        return ConstantRateTimeSeriesMetadata(\
            dtype = timeseries.dtype, 
            shape = timeseries.shape[1:],
            frequency = timeseries.frequency,
            duration = timeseries.duration)
    
    def _encode_timedelta(self, td):
        return (td.astype(np.uint64).tostring(), str(td.dtype)[-3:-1])

    def _decode_timedelta(self, t):
        if isinstance(t, np.timedelta64):
            return t
        
        v = np.fromstring(t[0], dtype = np.uint64)[0]
        s = t[1] 
        return np.timedelta64(long(v), s)
    
    def __repr__(self):
        return repr((
             str(np.dtype(self.dtype)), 
             self.shape,
             self._encode_timedelta(self.frequency),
             self._encode_timedelta(self.duration)
         ))

class BaseConstantRateTimeSeriesEncoder(Node):
    
    content_type = 'application/octet-stream'
    
    def __init__(self, needs = None):
        super(BaseConstantRateTimeSeriesEncoder, self).__init__(needs = needs)
        self.metadata = None
    
    def _prepare_data(self, data):
        raise NotImplementedError()
    
    def _process(self, data):
        data = self._prepare_data(data)
        if not self.metadata:
            self.metadata = ConstantRateTimeSeriesMetadata\
                .from_timeseries(data)
            packed = self.metadata.pack()
            yield packed
        
        encoded = data.tostring()
        yield encoded
        
class ConstantRateTimeSeriesEncoder(BaseConstantRateTimeSeriesEncoder):
    
    def _prepare_data(self, data):
        return data

class PackedConstantRateTimeSeriesEncoder(BaseConstantRateTimeSeriesEncoder):
    
    def __init__(self, needs = None, axis = 1):
        super(PackedConstantRateTimeSeriesEncoder, self).__init__(needs = needs)
        self.axis = axis
    
    def _prepare_data(self, data):
        packedbits = np.packbits(data.astype(np.uint8), axis = self.axis)
        
        return ConstantRateTimeSeries(\
          packedbits, 
          frequency = data.frequency,
          duration = data.duration)

def _np_from_buffer(b, shape, dtype, freq, duration):
    f = np.frombuffer if len(b) else np.fromstring
    shape = tuple(int(x) for x in shape)
    f = f(b, dtype = dtype).reshape(shape)
    return ConstantRateTimeSeries(f, freq, duration)

class GreedyConstantRateTimeSeriesDecoder(Decoder):
    
    def __init__(self):
        super(GreedyConstantRateTimeSeriesDecoder, self).__init__()

    def __call__(self,flo):
        metadata, bytes_read = ConstantRateTimeSeriesMetadata.unpack(flo)
        
        leftovers = flo.read()
        leftover_bytes = len(leftovers)
        first_dim = leftover_bytes / metadata.totalsize
        dim = (first_dim,) + metadata.shape
        out = _np_from_buffer(\
           leftovers,
           dim,
           metadata.dtype, 
           metadata.frequency, 
           metadata.duration)
        return out

    def __iter__(self,flo):
        yield self(flo)

class ConstantRateTimeSeriesFeature(Feature):
    
    def __init__(\
        self,
        extractor,
        needs = None,
        store = False,
        key = None,
        encoder = ConstantRateTimeSeriesEncoder,
        decoder = GreedyConstantRateTimeSeriesDecoder(),
        **extractor_args):
        
        super(ConstantRateTimeSeriesFeature ,self).__init__(\
            extractor,
            needs = needs,
            store = store,
            encoder = encoder,
            decoder = decoder,
            key = key,
            **extractor_args)
        
class ConstantRateTimeSeries(np.ndarray):
    '''
    A TimeSeries implementation with samples of a constant duration and 
    frequency.
    '''
    __array_priority__ = 10.0
    
    def __new__(cls, input_array, frequency, duration = None):
        if not isinstance(frequency, np.timedelta64):
            raise ValueError('duration must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = frequency.__class__))
        
        if duration != None and not isinstance(duration, np.timedelta64):
            raise ValueError('start must be of type {t} but was {t2}'.format(\
               t = np.timedelta64, t2 = duration.__class__))
        
        obj = np.asarray(input_array).view(cls)
        obj.frequency = frequency
        obj.duration = duration or frequency
        return obj

    def concatenate(self, other):
        if self.frequency == other.frequency and self.duration == other.duration:
            return ConstantRateTimeSeries(\
                  np.concatenate([self, other]),
                  frequency = self.frequency,
                  duration = self.duration)
        raise ValueError(\
         'self and other must have the same sample frequency and sample duration')

    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)
    
    @property
    def samplerate(self):
        return SampleRate(self.frequency, self.duration)
    
    @property
    def overlap(self):
        return self.samplerate.overlap
    
    @property
    def span(self):
        overlap = self.duration - self.frequency
        return TimeSlice((len(self) * self.frequency) + overlap) 

    def __array_finalize__(self, obj):
        if obj is None: return
        self.frequency = getattr(obj, 'frequency', None)
        self.duration = getattr(obj, 'duration', None)
        
    def __getitem__(self, index):
        if isinstance(index, TimeSlice):
            diff = self.duration - self.frequency
            start_index = \
                max(0, np.floor((index.start - diff) / self.frequency))
            stop_index = np.ceil(index.end / self.frequency)
            return self[start_index : stop_index]
        
        return super(ConstantRateTimeSeries, self).__getitem__(index)


        