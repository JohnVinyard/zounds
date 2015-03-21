
class TimeSeries(object):
    '''
    TimeSeries and derived-classes are meant to handle data samples over time
    in a completely generic way.
    
    At the moment, I can think of three types of data:
    
    static - zero sample rate, or a single sample value that applies to every
    moment in time
    
    constant sample rate- a sample is taken every n units of time 
    (frequency and duration)
    
    sparse - samples occur in a completely arbitrary way. Samples are not spaced
    evenly, and multiple samples may occur simultaneously.
     
    Various types of interpolation should be implemented, and each instance should
    make explicit whether interpolation is meaningful for the type of data.
    
    TimeSeries should be indexable with different units of time.  For constant
    rate series, the default unit should be the sampling rate of the data, so
    that normal list/array indexing can occur.
    
    For constant rate TimeSeries, all sampling rates should be expressed in 
    microseconds.
    '''
    def __init__(self):
        object.__init__(self)
 

class Static(TimeSeries):
    
    def __init__(self,value):
        TimeSeries.__init__(self)
        self.value = value

class ConstantRate(TimeSeries):
    
    def __init__(self,data,sr_seconds = 1.):
        TimeSeries.__init__(self)
        self.data = data
        self._sr = sr_seconds

class VariableRate(TimeSeries):
    
    def __init__(self,times,samples):
        TimeSeries.__init__(self)
        self.times = times
        self.samples = samples
        

