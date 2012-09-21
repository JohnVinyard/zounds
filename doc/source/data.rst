Data Persistence
=================================

.. currentmodule:: zounds.data

.. automodule:: zounds.data

Storing Computed Features
----------------------------------
This section will be useful to developers interested in implementing new storage
backends, or adding new functionality to zounds, but writing a standard zounds
application should only require you to interact with these classes in one place:
your :code:`config.py` file.  Here's an example of a simple configuration file::

	# import zounds' logging configuration so it can be used in this application
	from zounds.log import *
	
	# User Config
	source = 'myapp'
	
	# Audio Config
	class AudioConfig:
	    samplerate = 44100
	    windowsize = 2048
	    stepsize = 1024
	    window = None
	
	# FrameModel
	from zounds.model.frame import Frames, Feature
	from zounds.analyze.feature.spectral import FFT,BarkBands
	
	class FrameModel(Frames):
	    fft = Feature(FFT, store = False)
	    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
	
	
	# import models that will be used in this application
	from zounds.model.framesearch import ExhaustiveSearch
	from zounds.model.pipeline import Pipeline
	
	# import the storage backends that will be used to persist those models
	from zounds.data.frame import FileSystemFrameController
	from zounds.data.search import PickledSearchController
	from zounds.data.pipeline import PickledPipelineController
	
	# create a mapping from model class -> storage backend instance
	data = {
	    ExhaustiveSearch    : PickledSearchController(),
	    Pipeline            : PickledPipelineController()
	}
	
	
	from zounds.environment import Environment
	dbfile = 'datastore'
	
	# instantiating the Environment class "wires up" the entire application
	Z = Environment(
	                source,
	                # tell the environment about features-of-interest
	                FrameModel,
	                # tell the environment which storage backend will persist
	                # and retreive computed features
	                FileSystemFrameController,
	                # tell the environment about arguments needed to instantiate
	                # the frames storage backend
	                (FrameModel,dbfile),
	                # tell the environment about storage backends for storing
	                # learning pipelines and searches
	                data,
	                # tell the environment about audio settings
	                audio = AudioConfig)

In general, there will be a layer of abstraction between you and 
:py:class:`zounds.data.*` classes.  Most tasks won't require you to know about
the specific storage backend, or even know about the standard 
:py:class:`~zounds.data.frame.frame.FrameController` API.

To analyze some audio and insert it into the database using the 
:py:class:`~zounds.acquire.acquirer.DiskAcquirer` class::

	>>> da = DiskAcquirer('/path/to/some/audio')
	>>> da.acquire()

To fetch a pattern::
	
	>>> FrameModel['7eb8ad4caa294f558cb7d74807861c98']
	FrameModel(
		source = Test3,
		nframes = 344,
		zounds_id = 7eb8ad4caa294f558cb7d74807861c98,
		external_id = 6670,
		n_seconds = 8.01088435374)

To list all zounds ids::
	
	>>> FrameModel.list_ids()
	set(['730dc2b692fc40c5b191ba843a34cc62', '4ac8c74e3d034c9881bb885d1c56e7bc', 
	'11130026cb8e44b2b034558cdd7b0c08', '809f171adc674eea809cd0fbd26a9aba', 
	'b279a5ac9f194b36a027e99321a7793e', '971d87ad34114b0cbafb2e3104abe681', 
	'5fcaca500cd948f3923f968de45040ab'])

To access feature statistics::

	>>> FrameModel.bark.mean()
	array([ 13.90597343,  25.91770744,  31.96687126,  27.87793159 ...

To train and persist a :py:class:`~zounds.model.pipeline.Pipeline` and retrieve
it::

	>>> p = Pipeline('pipeline/bark_kmeans',PrecomputedFeature(1,FrameModel.bark),NoOp(),KMeans(10))
	>>> p.train(100, lambda : True) # train on 100 random examples and store the result
	>>> del p
	>>> Pipeline['pipeline/bark_kmeans']
	Pipeline(
		preprocess = NoOp(),
		learn = KMeans(n_centroids = 10),
		trained_date = 2012-09-21 22:17:09.707488,
		id = pipeline/bark_kmeans,
		fetch = PrecomputedFeature(feature = bark, nframes = 1))
	

Notice that in all these examples, data is being persisted to and retrieved from
some datastore, but you don't have to interface directly with any backend-specific
classes.

If you're still interested in the particulars of data backends, read on.

The FrameController API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: zounds.data.frame.frame

.. autoclass:: FrameController
	:members:

The PyTablesFrameController
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: zounds.data.frame.pytables

.. autoclass:: PyTablesFrameController
	:members: __init__

The FileSystemFrameController
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: zounds.data.frame.filesystem

.. autoclass:: FileSystemFrameController
	:members: __init__

Storing Learning Pipelines
----------------------------------
blah blah

Storing Search Indexes
----------------------------------
blah blah
