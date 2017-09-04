
class ChunkSizeBytes(object):
    """
    A convenience class to help describe a chunksize in bytes for the
    :class:`featureflow.ByteStream` in terms of audio sample batch sizes.

    Args:
        samplerate (SampleRate): The samples-per-second factor
        duration (numpy.timedelta64): The length of desired chunks in seconds
        channels (int): Then audio channels factor
        bit_depth (int): The bit depth factor

    Examples:
        >>> from zounds import ChunkSizeBytes, Seconds, SR44100
        >>> chunksize = ChunkSizeBytes(SR44100(), Seconds(30))
        >>> chunksize
        ChunkSizeBytes(samplerate=SR44100(f=2.2675736e-05, d=2.2675736e-05)...
        >>> int(chunksize)
        5292000

    """
    def __init__(self, samplerate, duration, channels=2, bit_depth=16):
        self.duration = duration
        self.bit_depth = bit_depth
        self.channels = channels
        self.samplerate = samplerate

    def __int__(self):
        byte_depth = self.bit_depth // 8
        total_samples = int(self.duration / self.samplerate.frequency)
        return int(total_samples * byte_depth * self.channels)

    def __repr__(self):
        msg = 'ChunkSizeBytes(samplerate={samplerate}, duration={duration}, ' \
              'channels={channels}, bit_depth={bit_depth})'

        return msg.format(
            samplerate=self.samplerate,
            duration=str(self.duration),
            channels=self.channels,
            bit_depth=self.bit_depth)
