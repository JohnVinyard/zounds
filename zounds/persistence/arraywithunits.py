from zounds.core import ArrayWithUnits
from zounds.persistence import DimensionEncoder, DimensionDecoder
from featureflow import Node, Decoder, Feature, NumpyMetaData
import struct
import json
import numpy as np


def _np_from_buffer(b, shape, dtype):
    f = np.frombuffer if len(b) else np.fromstring
    return f(b, dtype=dtype).reshape(shape)


class ArrayWithUnitsEncoder(Node):

    content_type = 'application/octet-stream'

    def __init__(self, needs=None):
        super(ArrayWithUnitsEncoder, self).__init__(needs=needs)
        self.encoder = DimensionEncoder()
        self.dimensions = None
        self.nmpy = None

    def _process(self, data):
        if self.dimensions is None:
            self.dimensions = data.dimensions
            d = list(self.encoder.encode(self.dimensions))
            encoded = json.dumps(d)
            yield struct.pack('I', len(encoded))
            yield encoded
        if self.nmpy is None:
            self.nmpy = NumpyMetaData(data.dtype, data.shape[1:])
            yield self.nmpy.pack()

        yield data.tostring()


class ArrayWithUnitsDecoder(Decoder):
    def __init__(self):
        super(ArrayWithUnitsDecoder, self).__init__()

    def __call__(self, flo):
        nbytes = struct.calcsize('I')
        json_len = struct.unpack('I', flo.read(nbytes))[0]
        d = json.loads(flo.read(json_len))
        decoder = DimensionDecoder()
        dimensions = list(decoder.decode(d))

        metadata, bytes_read = NumpyMetaData.unpack(flo)
        leftovers = flo.read()
        leftover_bytes = len(leftovers)
        first_dim = leftover_bytes / metadata.totalsize
        dim = (first_dim,) + metadata.shape
        raw = _np_from_buffer(leftovers, dim, metadata.dtype)

        return ArrayWithUnits(raw, dimensions)

    def __iter__(self, flo):
        yield self(flo)


class ArrayWithUnitsFeature(Feature):
    def __init__(
            self,
            extractor,
            needs=None,
            store=False,
            key=None,
            encoder=ArrayWithUnitsEncoder,
            decoder=ArrayWithUnitsDecoder(),
            **extractor_args):
        super(ArrayWithUnitsFeature, self).__init__(
                extractor,
                needs=needs,
                store=store,
                encoder=encoder,
                decoder=decoder,
                key=key,
                **extractor_args)
