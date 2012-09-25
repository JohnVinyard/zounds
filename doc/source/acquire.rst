Importing Audio
===============================

.. currentmodule:: zounds.acquire

.. automodule:: zounds.acquire

The ingest.py Script
------------------------------
If all you need to do is analyze audio files that live on your machine, and you
used the :role:`quickstart script <quick-start>` to start your application, there's
another script in your application directory called :code:`ingest.py`.  This
script provides a convenient command-line interface to the 
:py:class:`~zounds.acquire.acquirer.DiskAcquirer` class.  
To analyze a bunch of sounds on your machine, just run::
	
	python ingest.py --path /path/to/my/sounds

If you don't have any sounds laying around, run the script with no arguments... ::

	python ingest.py

... and a small set of sounds will be downloaded, unpacked, and analyzed.

The Acquirer Interface
------------------------------
.. currentmodule:: zounds.acquire.acquirer

.. autoclass:: Acquirer
	:members: source,_acquire,framemodel,framecontroller,acquire

The DiskAcquirer
------------------------------
.. autoclass:: DiskAcquirer
	:members: __init__