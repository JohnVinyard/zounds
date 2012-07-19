.. Zounds documentation master file, created by
   sphinx-quickstart on Fri Mar 23 10:54:41 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Zounds' documentation!
==================================
**Zounds** is a library designed to make experimenting with audio features a breeze!

- **Zounds allows you to define sets of features in an intuitive, pythonic way**::

	class FrameModel(Frames):
		fft = Feature(FFT, store = False)
		bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)

- **Zounds makes analyzing large collections of sound and storing the features easy**. It can be as simple as::

	python ingest.py --path /path/to/your/audio_collection

- **Zounds doesn't mind if you change yours**. If you modify your feature set, Zounds knows, and handles the dirty work of updating your data store.

- **Zounds implements some neat, unsupervised learning algorithms**, so you can learn good representations of your data, and use them as part of your feature set::
	 
	pl = Pipeline(
		'bark/rbm',
		# we're learning a representation of bark bands
		PrecomputedFeature(1,FrameModel.bark),
		# preprocess the data by subtracting the mean and dividing by the 
		# standard deviation, feature-wise
		MeanStd(),
		# use a restricted boltzmann machine to learn a representation of the 
		# data
		LinearRbm(100,500))
	
	# grab 10,000 samples of bark bands from the database, at random, and train
	# for 100 epochs
	pl.train(10000,lambda epoch,error: epoch > 100)
	


Contents:

.. toctree::
   :maxdepth: 2

Frames
===================================
.. automodule:: zounds.model.frame
	:members:

Pattern
===================================
.. automodule:: zounds.model.pattern
	:members:

Analyze
===================================
.. automodule:: zounds.analyze.audiostream
	:members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

