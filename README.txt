About
===============================================================================
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


Download
====================================
Get the latest source distribution here: `zounds-0.03.tar.gz <https://bitbucket.org/jvinyard/zounds2/downloads/zounds-0.03.tar.gz>`_.
The source distribution is the latest stable release, so this is the preferred way to get Zounds if you're planning to go through the :doc:`quickstart tutorial <quick-start>`, or write a client application.

Get the source: `Zounds on BitBucket <https://bitbucket.org/jvinyard/zounds2/src>`_.

Installation
=================================

=================================
Caveats
=================================
This is a very early release of Zounds. So far, it has only been tested on Ubuntu 10.10.

=================================
Setup.py
=================================
Run::

	sudo python setup.py install


Be sure to enable realtime scheduling for JACK.

.. NOTE::
	setup.py installs quite a few libraries and Python packages, and may take
	some time to run.

.. WARNING::
	During installation, `scikits.audiolab` causes an error that halts the 
	script.  You can simply re-issue the command above, and things will continue
	along just fine.  It's a bit klunky, but it gets the job done.

=================================
Test Audio
=================================
Zounds uses the `JACK <http://jackaudio.org/>`_ library to play audio.  Setup.py
added the user you're logged in as to the "audio" group, which gives you realtime
audio permissions. **You'll need to log out and back in for these changes to take
effect**. Once you do, run::

	zounds-audio-test.py

You should hear a rhythmic ticking sound. This means that everything is setup
properly.

=================================
Bravo!
=================================
You've succesfully installed zounds! Now on to the :doc:`Quick Start Tutorial </quick-start>`


	


