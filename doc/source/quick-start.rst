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
	
	
	# Data backends
	from zounds.model.framesearch import ExhaustiveSearch
	from zounds.model.pipeline import Pipeline
	from zounds.data.frame import PyTablesFrameController
	from zounds.data.search import PickledSearchController
	from zounds.data.pipeline import PickledPipelineController
	
	data = {
	    ExhaustiveSearch    : PickledSearchController(),
	    Pipeline            : PickledPipelineController()
	}
	
	
	from zounds.environment import Environment
	dbfile = 'datastore/frames.h5'
	Z = Environment(
	                source,                  # name of this application
	                FrameModel,              # our frame model
	                PyTablesFrameController, # FrameController class
	                (FrameModel,dbfile),     # FrameController args
	                data,                    # data-backend config
	                audio = AudioConfig)     # audio configuration     



That's a lot to take in, but the parts we're interested for now are the AudioConfig,
:py:class:`FrameModel <zounds.model.frame.Frames>`, and 
:py:class:`~zounds.environment.Environment` classes.

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
files with a different sample rate will be resampled prior to processing.  We'll
process windows of 2048 samples at a time, and each window will overlap with the
previous window by half.

-----------------------------------
FrameModel Class
-----------------------------------
The :py:class:`FrameModel <zounds.model.frame.Frames>` class is where you define 
the features you're interested in.  Some features will be intermediate, i.e., 
they're needed to compute other features down the line, but aren't interesting 
on their own.::
		
		class FrameModel(Frames):
		    fft = Feature(FFT)
		    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)

In this case, we're computing an :py:class:`~zounds.analyze.feature.spectral.FFT` 
on each window of audio, and then mapping the FFT coefficients onto the 
`Bark Scale <http://en.wikipedia.org/wiki/Bark_scale>`_.

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
	                data,                               # data-backend config
	                audio = AudioConfig)				# audio config
	                              

Other scripts in your application should have an import statement like this... ::
	
	from config import *

...near the top, so that everything will be wired up correctly.

=====================================================
Importing Audio
=====================================================
Let's analyze some audio! Use the following command... ::

	python ingest.py

...to download a small set of pre-selected sounds and process them, or run... ::
	
	python ingest.py --path /path/to/my/sounds

...to process a folder full of sounds on your machine.  If you don't have any audio 
files laying around, `Freesound.org <http://www.freesound.org>`_ is highly
recommended!

.. WARNING::
	Keep in mind that Zounds can't handle mp3 files yet.  Mp3 files will be skipped by ingest.py.

=====================================================
Visualize the Results
=====================================================
Let's make sure that the analysis worked. Type::

	python display.py

This will create a simple html file with images of the features we just computed.
Use your favorite browser to view the results like so::

	google-chrome display/index.html

====================================================
Change Your FrameModel
====================================================
Let's add some new features. Open up config.py in your favorite text editor, and
change the :py:class:`FrameModel <zounds.model.frame.Frames>` portion so it looks 
like this::

	from zounds.model.frame import Frames, Feature
	from zounds.analyze.feature.spectral import FFT,BarkBands,SpectralCentroid,SpectralFlatness,Loudness
	from zounds.analyze.feature.composite import Composite
	
	# Here's where we define the features we're interested in.
	class FrameModel(Frames):
	    fft = Feature(FFT, store = False)
	    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
	    loud = Feature(Loudness, needs = bark)
	    centroid = Feature(SpectralCentroid, needs = bark)
	    flat = Feature(SpectralFlatness, needs = bark)
	    vec = Feature(Composite, needs = [centroid,flat])

Here, we've added four new features

- **Loudness** measures the amplitude of the sound
- **SpectralCentroid** measures the center of gravity of the spectrum, or how perceptually "high" or "low" a frame sounds.
- **SpectralFlatness** measures how noisy a frame sounds.  Imagine this as the scale between a pure sine tone and white noise.
- **Composite** combines the previous two scalar features into a single, two-dimensional feature

Save the file. Now, the next time we try to do anything in our app, the changes
in our feature set will be detected, and the datastore will be updated to reflect
those changes.
 
Run... ::

	python display.py

...again. You should see some indication that your database is being upgraded.  
Take a look at the results again, e.g.... ::

	google-chrome display/index.html

... and you should see that the new features have been computed.

Zounds does its best to perform the update in the most efficient way possible, 
so, in this case, the :code:`fft` and :code:`bark` features were not recomputed.
The stored :code:`bark` values were passed along to the new features.

====================================================
Do a Search
====================================================
Zounds was designed to make experimenting with different features for audio 
similarity search as painless as possible.  There's a file called search.py in 
the myapp folder, which will perform searches using pre-computed features in your 
database.  Let's give it a shot.::

	python search.py --feature vec --searchclass ExhaustiveSearch --sounddir /path/to/audio_folder --nresults 2

Here's a quick explanation of the options:

- **feature** determines which feature we'll use to compare segments of sound
- **searchclass** determines which instance of a zounds.model.framesearch.FrameSearch-derived class we'll be using.  
  ExhaustiveSearch performs a brute force search with no indexing.
- **sounddir** is a directory containing audio files from which we'll be randomly pulling queries
- **nresults** is the number of results we'd like returned for each query.  We've chosen a low number here, since our database is probably pretty small.

Chances are the search results won't impress you much, since we're using very 
low-level features, but this should give you a feel for how to quickly try out
other features and search implementations.

====================================================
The FrameModel class
====================================================
Let's see what the FrameModel class you defined in config.py is good for.  Start
an interactive python session, and let's play around a bit.

First, grab a random sound from the database::

	>>> from config import FrameModel,Z
	>>> frames = FrameModel.random()
	>>> frames
	FrameModel(
		source = sound,
		nframes = 1064,
		zounds_id = 10e6d221ea194efc90f2ca95c1ea7551,
		external_id = 32079,
		n_seconds = 24.7292517007)

Let's check out some statistics of the computed features::

	>>> FrameModel.bark.mean()
	array([ 18.13603592,  29.5746994 ,  25.22009659,  19.1783886 ,...
	>>> FrameModel.bark.std()
	array([ 26.31206703,  42.49121475,  34.20608139,  24.72426033,...
	>>> FrameModel.loud.min()
	0.0
	>>> FrameModel.loud.max()
	6146.4766

Now, let's play the sound

	>>> Z.play(frames.audio)

If it's a longer sound, and you're tired of listening, just hit ctl-c.

Features are just numpy arrays::
	
	>>> frames.bark.shape
	(1064, 100)
	>>> frames.bark.dtype
	dtype('float32')

Feature's that aren't stored can be computed on the fly and cached for the lifetime
of the :py:class:`~zounds.model.frame.Frames`-derived instance::

	>>> frames.fft
	array([[  1.80337372e-06,   8.16792412e-06,   2.81055575e-05, ...,
	>>> frames.fft.shape
	(1066, 1024)
	>>> frames.fft.dtype
	dtype('float64')

Since we've computed a loudness value for every frame, we can reorder the frames
and play them from quietest to loudest::

	>>> import numpy as np
	>>> l = np.argsort(frames.loud)
	>>> Z.play(frames.audio[l])

How about playing the sound so it goes from the most to least "bright" ::

	>>> c = np.argsort(frames.centroid)[::-1]
	>>> Z.play(frames.audio[c])

Or from least to most noisy ::

	 >>> f = np.argsort(frames.flatness)
	 >>> Z.play(frames.audio[f])
	
	
	



