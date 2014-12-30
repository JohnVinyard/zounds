import unittest2
from collections import defaultdict

from extractor import NotEnoughData
from model import BaseModel
from feature import Feature,JSONFeature
from dependency_injection import Registry
from data import *
from util import chunked

data_source = {
	'mary'   : 'mary had a little lamb little lamb little lamb',
	'humpty' : 'humpty dumpty sat on a wall humpty dumpty had a great fall',
	'numbers' : range(10),
	'cased' : 'This is a test.'
}

class TextStream(Node):
	
	def __init__(self, needs = None):
		super(TextStream,self).__init__(needs = needs)
	
	def _process(self,data):
		flo = StringIO(data_source[data])
		for chunk in chunked(flo,chunksize = 3):
			yield chunk

class ToUpper(Node):

	def __init__(self, needs = None):
		super(ToUpper,self).__init__(needs = needs)

	def _process(self,data):
		yield data.upper()

class ToLower(Node):

	def __init__(self, needs = None):
		super(ToLower,self).__init__(needs = needs)

	def _process(self,data):
		yield data.lower()

class Concatenate(Node):

	def __init__(self, needs = None):
		super(Concatenate,self).__init__(needs = needs)
		self._cache = defaultdict(str)

	def _enqueue(self,data,pusher):
		self._cache[id(pusher)] += data

	def _dequeue(self):
		if not self._finalized:
			raise NotEnoughData()

		return self._cache

	def _process(self,data):
		s = ''
		for v in data.itervalues():
			s += v
		yield s

class NumberStream(Node):

	def __init__(self, needs = None):
		super(NumberStream,self).__init__(needs = needs)

	def _process(self,data):
		l = data_source[data]
		for i in xrange(0,len(l),3):
			yield l[i : i + 3]

class Add(Node):

	def __init__(self, rhs = 1, needs = None):
		super(Add,self).__init__(needs = needs)
		self._rhs = rhs

	def _process(self,data):
		yield [c + self._rhs for c in data]

class SumUp(Node):

	def __init__(self, needs = None):
		super(SumUp,self).__init__(needs = needs)
		self._cache = dict()

	def _enqueue(self,data,pusher):
		self._cache[id(pusher)] = data

	def _dequeue(self):
		if len(self._cache) != len(self._needs):
			raise NotEnoughData()
		v = self._cache
		self._cache = dict()
		return v

	def _process(self,data):
		results = [str(sum(x)) for x in zip(*data.itervalues())]
		yield ''.join(results)

class Tokenizer(Node):
	
	def __init__(self, needs = None):
		super(Tokenizer,self).__init__(needs = needs)
		self._cache = ''

	def _enqueue(self,data,pusher):
		self._cache += data

	def _finalize(self,pusher):
		self._cache += ' '

	def _dequeue(self):
		last_index = self._cache.rfind(' ')
		if last_index == -1:
			raise NotEnoughData()
		current = self._cache[:last_index + 1]
		self._cache = self._cache[last_index + 1:]
		return current
	
	def _process(self,data):
		yield filter(lambda x : x, data.split(' '))

class WordCount(Node):
	
	def __init__(self, needs = None):
		super(WordCount,self).__init__(needs = needs)
		self._cache = defaultdict(int)

	def _enqueue(self,data,pusher):
		for word in data:
			self._cache[word.lower()] += 1

	def _dequeue(self):
		if not self._finalized:
			raise NotEnoughData()

		return super(WordCount,self)._dequeue()

class Document(BaseModel):
	
	stream = Feature(TextStream, store = True)
	words  = Feature(Tokenizer, needs = stream, store = False)
	count  = JSONFeature(WordCount, needs = words, store = False)

class Document2(BaseModel):
	
	stream = Feature(TextStream, store = False)
	words  = Feature(Tokenizer, needs = stream, store = False)
	count  = JSONFeature(WordCount, needs = words, store = True)

class Numbers(BaseModel):

	stream = Feature(NumberStream,store = False)
	add1 = Feature(Add, needs = stream, store = False, rhs = 1)
	add2 = Feature(Add, needs = stream, store = False, rhs = 1)
	sumup = Feature(SumUp, needs = [add1,add2], store = True)

class Doc3(BaseModel):

	stream = Feature(TextStream, store = True)
	uppercase = Feature(ToUpper, needs = stream, store = True)
	lowercase = Feature(ToLower, needs = stream, store = False)
	cat = Feature(Concatenate, needs = [uppercase,lowercase], store = False)

class IntegrationTest(unittest2.TestCase):

	def setUp(self):
		Registry.register(IdProvider,UuidProvider())
		Registry.register(KeyBuilder,StringDelimitedKeyBuilder())
		Registry.register(Database,InMemoryDatabase())
		Registry.register(DataWriter,DataWriter)
		Registry.register(DataReader,DataReaderFactory())

	def test_can_process_and_retrieve_stored_feature(self):
		_id = Document.process(stream = 'mary')
		doc = Document(_id)
		self.assertEqual(data_source['mary'],doc.stream.read())

	def test_can_correctly_decode_feature(self):
		_id = Document2.process(stream = 'mary')
		doc = Document2(_id)
		self.assertTrue(isinstance(doc.count,dict))

	def test_can_retrieve_unstored_feature_when_dependencies_are_satisfied(self):
		_id = Document.process(stream = 'humpty')
		doc = Document(_id)
		d = doc.count
		self.assertEqual(2,d['humpty'])
		self.assertEqual(1,d['sat'])

	def test_cannot_retrieve_unstored_feature_when_dependencies_are_not_satisfied(self):
		_id = Document2.process(stream = 'humpty')
		doc = Document2(_id)
		self.assertRaises(AttributeError,lambda : doc.stream)

	def test_feature_with_multiple_inputs(self):
		_id = Numbers.process(stream = 'numbers')
		doc = Numbers(_id)
		self.assertEqual('2468101214161820',doc.sumup.read())

	def test_unstored_feature_with_multiple_inputs_can_be_computed(self):
		_id = Doc3.process(stream = 'cased')
		doc = Doc3(_id)
		self.assertEqual('this is a test.THIS IS A TEST.',doc.cat.read())
