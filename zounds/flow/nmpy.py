import numpy as np
from extractor import Node
from feature import Feature
from util import chunked
import struct

class NumpyMetaData(object):

	def __init__(self,dtype = None,shape = None):
		self.dtype = np.uint8 if dtype is None else dtype
		self.shape = shape or ()
	
	@property
	def itemsize(self):
		return np.dtype(self.dtype).itemsize
	
	@property
	def size(self):
		return np.product(self.shape)
	
	@property
	def totalsize(self):
		return self.itemsize * self.size
	
	def __getitem__(self,index):
		if not isinstance(index,slice):
			raise ValueError('index must be a slice instance')
		
		return NumpyMetaData(self.dtype,self.shape[index])

	def __repr__(self):
		return repr((str(np.dtype(self.dtype)),self.shape))

	def __str__(self):
		return self.__repr__()

	def pack(self):
		s = str(self)
		l = len(s)
		return struct.pack('B{n}s'.format(n = l),l,s)

	@classmethod
	def unpack(cls,flo):
		l = struct.unpack('B',flo.read(1))[0]
		bytes_read = 1 + l
		return cls(*eval(flo.read(l))),bytes_read

class NumpyEncoder(Node):
	
	content_type = 'application/octet-stream'
	
	def __init__(self, needs = None):
		super(NumpyEncoder,self).__init__(needs = needs)
		self.metadata = None
	
	def _process(self,data):
		if not self.metadata:
			self.metadata = NumpyMetaData(\
				dtype = data.dtype, shape = data.shape[1:])
			yield self.metadata.pack()
		
		yield data.tostring()


def _np_from_buffer(b,shape,dtype):
	f = np.frombuffer if len(b) else np.fromstring
	return f(b,dtype = dtype).reshape(shape)

class GreedyNumpyDecoder(Node):

	def __init__(self,needs = None):
		super(GreedyNumpyDecoder,self).__init__(needs = needs)

	def __call__(self,flo):
		metadata,_ = NumpyMetaData.unpack(flo)
		leftovers = flo.read()
		leftover_bytes = len(leftovers)
		first_dim = (leftover_bytes / (metadata.size * metadata.itemsize))
		dim = (first_dim,) + metadata.shape
		return _np_from_buffer(leftovers,dim,metadata.dtype)

	def __iter__(self,flo):
		yield self(flo)

class StreamingNumpyDecoder(Node):
	
	def __init__(self,needs = None,n_examples = 100):
		super(StreamingNumpyDecoder,self).__init__(needs = needs)
		self.n_examples = n_examples
	
	def __call__(self,flo):
		return self.__iter__(flo)

	def __iter__(self,flo):
		metadata,_ = NumpyMetaData.unpack(flo)
		example_size = metadata[1:].totalsize
		chunk_size = int(example_size * self.n_examples)
		count = 0
		
		for chunk in chunked(flo,chunk_size):
			n_examples = len(chunk) // example_size
			yield _np_from_buffer(\
				chunk,
				(n_examples,) + metadata[1:].shape,
				metadata.dtype)
			count += 1
		
		if count == 0:
			yield _np_from_buffer(\
				buffer(''),
				(0,) + metadata[1:].shape,
				metadata.dtype)
		
class NumpyFeature(Feature):
	
	def __init__(\
		self,
		extractor,
		needs = None,
		store = False,
		key = None,
		decoder = GreedyNumpyDecoder(),
		**extractor_args):
		
		super(NumpyFeature,self).__init__(\
		    extractor,
		    needs = needs,
		    store = store,
		    encoder = NumpyEncoder,
		    decoder = decoder,
		    key = key,
		    **extractor_args)
