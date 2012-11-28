About
====================================

**Zounds** is a library designed to make experimenting with audio features a breeze!

- **Zounds allows you to define sets of features in an intuitive, pythonic way**::

	class FrameModel(Frames):
		fft = Feature(FFT, store = False)
		bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
		loudness = Feature(Loud, needs = bark)
		centroid = Feature(SpectralCentroid, needs = bark)
		flatness = Feature(SpectralFlatness, needs = bark)

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
	# for 100 epochs. Then save the results.
	pl.train(10000,lambda epoch,error: epoch > 100)

Install
====================================

1. Get the tarball of the latest stable release from the `downloads page <https://bitbucket.org/jvinyard/zounds2/downloads>`_, or, clone the repository.
2. Run ``setup.py``.  This will install quite a few libraries and other dependencies, and may take some time to run. **Note that the script will halt when trying to install ``scikits.audiolab`` the first time.  Simply re-issuing the ``setup.py`` command will get things back on track**. 

Documentation
====================================

Go through the `quickstart tutorial <http://johnvinyard.com/zoundsdoc/quick-start.html>`_, and then check out the `API Documentation <http://johnvinyard.com/zoundsdoc/api.html>`_.


