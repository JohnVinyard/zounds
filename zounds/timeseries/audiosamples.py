from samplerate import AudioSampleRate, audio_sample_rate
from soundfile import SoundFile
from io import BytesIO
from zounds.core import IdentityDimension, ArrayWithUnits
from timeseries import TimeDimension, TimeSlice
from duration import Picoseconds
from samplerate import SampleRate
import numpy as np


class AudioSamples(ArrayWithUnits):
    """
    `AudioSamples` represents constant-rate samples of a continuous audio signal
    at common sampling rates.

    It is a special case of an :class:`~zounds.core.ArrayWithUnits` whose first
    dimension is a :class:`~zounds.timeseries.TimeDimension` that has a common
    audio sampling rate (e.g. :class:`~zounds.timeseries.SR44100`).

    Args:
        array (np.ndarray): The raw sample data
        samplerate (SampleRate): The rate at which data was sampled

    Raises:
        ValueError: When array has a second dimension with size greater than 2
        TypeError: When samplerate is not a
            :class:`~zounds.timeseries.AudioSampleRate`
            (e.g. :class:`~zounds.timeseries.SR22050`)

    Examples::
        >>> from zounds import AudioSamples, SR44100, TimeSlice, Seconds
        >>> import numpy as np
        >>> raw = np.random.normal(0, 1, 44100*10)
        >>> samples = AudioSamples(raw, SR44100())
        >>> samples.samples_per_second
        44100
        >>> samples.channels
        1
        >>> sliced = samples[TimeSlice(Seconds(2))]
        >>> sliced.shape
        (88200,)
    """

    def __new__(cls, array, samplerate):
        if array.ndim == 1:
            dimensions = [TimeDimension(*samplerate)]
        elif array.ndim == 2:
            dimensions = [TimeDimension(*samplerate), IdentityDimension()]
        else:
            raise ValueError(
                'array must be one (mono) or two (multi-channel) dimensions')

        if not isinstance(samplerate, AudioSampleRate):
            raise TypeError('samplerate should be an AudioSampleRate instance')

        return ArrayWithUnits.__new__(cls, array, dimensions)

    def __add__(self, other):
        try:
            if self.samplerate != other.samplerate:
                raise ValueError(
                    'Samplerates must match, but they were '
                    '{self.samplerate} and {other.samplerate}'
                        .format(**locals()))
        except AttributeError:
            pass
        return super(AudioSamples, self).__add__(other)

    def kwargs(self):
        return {'samplerate': self.samplerate}

    def sum(self, axis=None, dtype=None, **kwargs):
        result = super(AudioSamples, self).sum(axis, dtype, **kwargs)
        if self.ndim == 2 and axis == 1:
            return AudioSamples(result, self.samplerate)
        else:
            return result

    @classmethod
    def from_file(cls, file_like_object):
        with SoundFile(file_like_object, mode='r') as f:
            samples = f.read(dtype=np.float32)
            return AudioSamples(samples, audio_sample_rate(f.samplerate))

    @classmethod
    def silence(cls, samplerate, duration, dtype=np.float32, channels=1):
        shape = (int(duration / samplerate.frequency), channels)
        silence = np.zeros(shape, dtype=dtype).squeeze()
        return cls(silence, samplerate)

    def silence_like(self, duration):
        x = self.__class__.silence(
            self.samplerate, duration, self.dtype, self.channels)
        x[:] = 1
        return x

    def pad_with_silence(self, silence_duration):
        silence = self.__class__.silence(
            self.samplerate, silence_duration, self.dtype)
        return AudioSamples(np.concatenate([self, silence]), self.samplerate)

    @property
    def samples_per_second(self):
        return int(Picoseconds(int(1e12)) / self.frequency)

    @property
    def duration_in_seconds(self):
        return self.duration / Picoseconds(int(1e12))

    @property
    def samplerate(self):
        return SampleRate(self.frequency, self.duration)

    @property
    def overlap(self):
        return self.samplerate.overlap

    @property
    def span(self):
        return self.dimensions[0].span

    @property
    def end(self):
        return self.dimensions[0].end

    @property
    def frequency(self):
        return self.dimensions[0].frequency

    @property
    def duration(self):
        return self.dimensions[0].duration

    @classmethod
    def from_example(cls, arr, example):
        return cls(arr, example.samplerate)

    @property
    def channels(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]

    @property
    def samplerate(self):
        return audio_sample_rate(self.samples_per_second)

    @property
    def mono(self):
        """
        Return this instance summed to mono.  If the instance is already mono,
        this is a no-op.
        """
        if self.channels == 1:
            return self
        x = self.sum(axis=1) * 0.5
        y = x * 0.5
        return AudioSamples(y, self.samplerate)

    @property
    def stereo(self):
        if self.channels == 2:
            return self
        return AudioSamples(np.vstack([self, self]).T, self.samplerate)

    def __getitem__(self, item):
        sliced = super(AudioSamples, self).__getitem__(item)
        try:
            if sliced.dimensions == self.dimensions:
                return AudioSamples(sliced, self.samplerate)
        except AttributeError:
            pass
        return sliced

    def sliding_window(self, samplerate, padding=True):
        ws = TimeSlice(duration=samplerate.duration)
        ss = TimeSlice(duration=samplerate.frequency)
        _, windowed = self.sliding_window_with_leftovers(ws, ss, dopad=padding)
        return windowed

    def encode(self, flo=None, fmt='WAV', subtype='PCM_16'):
        """
        Return audio samples encoded as bytes given a particular audio format

        Args:
            flo (file-like): A file-like object to write the bytes to.  If flo
                is not supplied, a new :class:`io.BytesIO` instance will be
                 created and returned
            fmt (str): A libsndfile-friendly identifier for an audio encoding
                (detailed here: http://www.mega-nerd.com/libsndfile/api.html)
            subtype (str): A libsndfile-friendly identifier for an audio
                encoding subtype (detailed here:
                http://www.mega-nerd.com/libsndfile/api.html)

        Examples:
            >>> from zounds import SR11025, AudioSamples
            >>> import numpy as np
            >>> silence = np.zeros(11025*10)
            >>> samples = AudioSamples(silence, SR11025())
            >>> bio = samples.encode()
            >>> bio.read(10)
            'RIFFx]\\x03\\x00WA'
        """
        flo = flo or BytesIO()
        with SoundFile(
                flo,
                mode='w',
                channels=self.channels,
                format=fmt,
                subtype=subtype,
                samplerate=self.samples_per_second) as f:
            f.write(self)
        flo.seek(0)
        return flo
