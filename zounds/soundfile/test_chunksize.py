import unittest2
from .chunksize import ChunkSizeBytes
from zounds.timeseries import SR44100, Seconds


class ChunkSizeBytesTests(unittest2.TestCase):
    def test_can_convert_to_integer_number_of_bytes(self):
        cs = ChunkSizeBytes(SR44100(), Seconds(30), channels=2, bit_depth=16)
        self.assertEqual(5292000, int(cs))

    def test_can_repr(self):
        cs = ChunkSizeBytes(SR44100(), Seconds(30), channels=2, bit_depth=16)
        s = cs.__repr__()
        self.assertEqual(
            'ChunkSizeBytes(samplerate=SR44100(f=2.2675736e-05, '
            'd=2.2675736e-05), duration=30 seconds, channels=2, bit_depth=16)',
            s)
