"""
The timeseries module introduces classes for dealing with time as it relates
to audio signals
"""

from duration import \
    Hours, Minutes, Seconds, Milliseconds, Microseconds, Picoseconds, \
    Nanoseconds

from audiosamples import AudioSamples

from samplerate import \
    SR11025, SR16000, SR22050, SR44100, SR48000, SR96000, HalfLapped, \
    audio_sample_rate,Stride, SampleRate, nearest_audio_sample_rate

from timeseries import TimeSlice, TimeDimension

from variablerate import \
    VariableRateTimeSeries, VariableRateTimeSeriesFeature, \
    VariableRateTimeSeriesEncoder, VariableRateTimeSeriesDecoder

from constantrate import ConstantRateTimeSeries

from functional import categorical, inverse_categorical