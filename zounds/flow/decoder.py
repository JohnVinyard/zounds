'''
decoders should be a callables that take a file-like object and return...anything
'''
import simplejson
from util import chunked
from extractor import Node

class Decoder(object):
	'''
	The simplest possible decoder takes a file-like object and returns it
	'''
	def __init__(self):
		super(Decoder,self).__init__()

	def __call__(self,flo):
		return flo

	def __iter__(self,flo):
		for chunk in chunked(flo):
			yield chunk

class GreedyDecoder(Decoder):
	'''
	A decoder that reads the entire file contents into memory
	'''
	def __init__(self):
		super(GreedyDecoder,self).__init__()

	def __call__(self,flo):
		return flo.read()

	def __iter__(self,flo):
		yield self(flo)

class JSONDecoder(GreedyDecoder):
	'''
	A decoder that interprets the data as JSON
	'''
	def __init__(self):
		super(JSONDecoder,self).__init__()

	def __call__(self,flo):
		return simplejson.loads(super(JSONDecoder,self).__call__(flo))

	def __iter__(self,flo):
		yield self(flo)

class DecoderNode(Node):

	def __init__(self,needs = None,decodifier = None):
		super(DecoderNode,self).__init__(needs = needs)
		self.decoder = decodifier

	def _process(self,data):
		for x in self.decoder.__iter__(data):
			yield x