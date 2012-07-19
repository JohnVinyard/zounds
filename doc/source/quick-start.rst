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