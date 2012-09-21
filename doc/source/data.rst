Data Persistence
=================================

.. currentmodule:: zounds.data

.. automodule:: zounds.data

Storing Computed Features
----------------------------------
This section will be useful to developers interested in implementing new storage
backends, or adding new functionality to zounds, but writing a standard zounds
application should only require you to interact with these classes in one place:
your :code:`config.py` file that defines your audio settings, feature set, and
storage backends.  Here's an example of a simple configuration file::

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
:py:class:`~zounds.data.frame.frame.FrameController` API, for example::

	lambda : True

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
