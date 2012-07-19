Quick Start
===================================

===============================
Setting Up Your Application
===============================
The setup.py script installed a command-line application you can use to get
up and running quickly.  Decide where you'd like your first application to live,
and then run::

	zounds-quickstart.py --directory myapp --source myapp

The script will create the myapp directory for you, and will place some useful
python scripts in it.

===============================
The Configuration File
===============================
Change directory into the new myapp folder, and open up config.py in your favorite
text editor. You'll see something like this::

	# The name of your application
	source = 'myapp'
	
	
	class AudioConfig:
	    samplerate = 44100
	    windowsize = 2048
	    stepsize = 1024
	    window = None
	
	from zounds.model.frame import Frames, Feature
	from zounds.analyze.feature.spectral import FFT,BarkBands
	
	# Here's where we define the features we're interested in.
	class FrameModel(Frames):
	    fft = Feature(FFT, store = False)
	    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
	
	# Data backends
	from zounds.model.framesearch import ExhaustiveSearch
	from zounds.data.frame import PyTablesFrameController
	from zounds.data.search import PickledSearchController
	
	data = {
	    ExhaustiveSearch    : PickledSearchController()
	}
	
	
	from zounds.environment import Environment
	dbfile = 'datastore/frames.h5'
	# setup the environment for our Zounds application
	Z = Environment(
	                source,                             # name of this application
	                FrameModel,                         # our frame model
	                PyTablesFrameController,            # FrameController class
	                (FrameModel,dbfile),                # FrameController args
	                data,                                # data-backend config
	                audio = AudioConfig)

That's a lot to take in, but the parts we're interested for now are the AudioConfig,
FrameModel, and Environment classes.

------------------------------------
Audio Configuration
------------------------------------
The AudioConfig class can be called anything. It just needs to have some attributes
which define how this zounds application will handle audio::

	class AudioConfig:
	    samplerate = 44100
	    windowsize = 2048
	    stepsize = 1024
	    window = None

This application's default sample rate will be 44100 hertz. Any incoming sound
files with a different sample rate will be resample prior to processing.  We'll
process windows of 2048 samples at a time, and each window will overlap with the
previous window by half.

-----------------------------------
FrameModel Class
-----------------------------------
The FrameModel class is where you define the features you're interested in.  Some
features will be intermediate, i.e., they're needed to compute other features down
the line, but aren't interesting on their own.::
		
		class FrameModel(Frames):
		    fft = Feature(FFT, store = False)
		    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)

In this case, we're computing an FFT on each window of audio, and then mapping
the FFT coefficients onto the `Bark Scale <http://en.wikipedia.org/wiki/Bark_scale>`_.
Note that features are stored by default, so we're saying explicitly that we don't
want to store the FFT features.

----------------------------------
The Zounds Environment
----------------------------------
Finally, we're setting everything up::

	from zounds.environment import Environment
	dbfile = 'datastore/frames.h5'
	# setup the environment for our Zounds application
	Z = Environment(
	                source,                             # name of this application
	                FrameModel,                         # our frame model
	                PyTablesFrameController,            # FrameController class
	                (FrameModel,dbfile),                # FrameController args
	                data,                                # data-backend config
	                audio = AudioConfig)
	                              

Other scripts in your application should have an import statement like this::
	
	from config import *

near the top, so that everything will be wired up correctly.

=====================================================
Importing Audio
=====================================================
Let's analyze some audio! Use the following command, replacing the --path option
with the path to a directory on your computer that contains some audio files::

	python ingest.py --path /path/to/audio_folder

If you don't have any audio files laying around, I highly recommend `Freesound.org <http://www.freesound.org>`_.

.. WARNING::
	Keep in mind that Zounds can't handle mp3 files yet.  Mp3 files will be skipped by ingest.py.

=====================================================
Visualize the Results
=====================================================
Let's make sure that the analysis worked. Type::

	python display.py display

This will create a simple html file with images of the features we just computed.
Use your favorite browser to view the results like so::

	google-chrome display/index.html

====================================================
Change Your FrameModel
====================================================
Let's add some new features. Open up config.py in your favorite text editor, and
change the FrameModel portion so it looks like this::

	from zounds.model.frame import Frames, Feature
	from zounds.analyze.feature.spectral import FFT,BarkBands,Loudness,SpectralCentroid,SpectralFlatness
	
	# Here's where we define the features we're interested in.
	class FrameModel(Frames):
	    fft = Feature(FFT, store = False)
	    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
	    loud = Feature(Loudness, needs = bark)
	    centroid = Feature(SpectralCentroid, needs = bark)
	    flat = Feature(SpectralFlatness, needs = bark)

Here, we've added three new features

- **Loudness** measures how loud a frame is.
- **SpectralCentroid** measures the center of gravity of the spectrum, or how perceptually "high" or "low" a frame sounds.
- **SpectralFlatness** measures how noisy a frame sounds.  Imagine this as the scale between a pure sine tone and white noise.

Save the file. Now, the next time we try to do anything in our app, the changes
will be detected, and the datastore will be updated to reflect our changes. Let's run::

	python display.py display

again. You should see some indication that your database is being upgraded.  Take
a look at the results again, e.g.::

	google-chrome display/index.html

and you should see that the new features have been computed.

====================================================
Do a Search
====================================================
Zounds was designed to make experimenting with different features for audio similarity
search as painless as possible.  There's a file called search.py in the myapp folder,
which will perform searches using precomputed features in your database.  Let's give
it a shot.::

	python search.py --feature bark --searchclass ExhaustiveSearch --sounddir /path/to/audio_folder --nresults 2

Here's a quick explanation of the options:

- **feature** determines which feature we'll use to compare segments of sound
- **searchclass** determines which instance of a zounds.model.framesearch.FrameSearch-derived class we'll be using.  
  ExhaustiveSearch performs a brute force search with no indexing.
- **sounddir** is a directory containing audio files from which we'll be randomly pulling queries
- **nresults** is the number of results we'd like returned for each query.  We've chosen a low number here, since our database is probably pretty small.

Chances are the search results won't impress you much, since we're using a very 
low-level feature, but this should give you a feel for how to quickly try out
other features and search implementations.

====================================================
The FrameModel class
====================================================
Let's see what the FrameModel class you defined in config.py is good for.::
	
	from config import FrameModel,Z
	import numpy as np
	print '================================================================'
	print 'The database-wide, feature-wise mean and standard deviation of the bark feature'
	print FrameModel.bark.mean()
	print FrameModel.bark.std()
	print '================================================================'
	print 'The database-wide min and max loudness values'
	print FrameModel.loud.min()
	print FrameModel.loud.max()
	print '================================================================'
	print 'Grab a random sound from the database and play it'
	frames = FrameModel.random()
	print frames
	Z.play(frames.audio)
	print '================================================================'
	print 'Features are just numpy arrays.  Here\'s the shape and datatype of the "loud" feature'
	print frames.loud.shape
	print frames.loud.dtype
	print '================================================================'
	print 'Features that aren\'t stored can be computed on the fly and cached by simply accessing them. Here\'s the shape and datatype of the "fft" feature'
	print frames.fft.shape
	print frames.fft.dtype
	print '================================================================'
	print 'playing the sound\'s frames, from quietest to loudest'
	li = np.argsort(frames.loud)
	Z.play(frames.audio[li])
	print '================================================================'
	print 'playing the sound\'s frames, from lowest to highest'
	ci = np.argsort(frames.centroid)
	Z.play(frames.audio[ci])
	print '================================================================'
	print 'playing the sound\'s frames, from least to most noisy'
	fi = np.argsort(frames.flat)
	Z.play(frames.audio[fi])
	
	
	



