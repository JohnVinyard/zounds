from timeseries import ConstantRateTimeSeries
from samplerate import AudioSampleRate


class AudioSamples(ConstantRateTimeSeries):

    def __new__(cls, array, samplerate):
        if not isinstance(samplerate, AudioSampleRate):
            raise TypeError('samplerate should be an AudioSampleRate instance')
        return ConstantRateTimeSeries.__new__(
            cls, array, samplerate.frequency, samplerate.duration)
