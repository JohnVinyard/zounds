'''
The :py:mod:`zounds.acquire` module defines classes that make it easy to import
audio data from the outside world into your zounds application.

For the moment, there's only a single implementation of the 
:py:class:`~zounds.acquire.acquirer.Acquirer` interface, the 
:py:class:`~zounds.acquire.acquirer.DiskAcquirer` class, which imports all the
audio files from a single directory on your local machine into your zounds
data store, however, there are plans for acquirers which use the 
`FreeSound.org <http://www.freesound.org/docs/api/>`_ and 
`SoundCloud <http://developers.soundcloud.com/docs>`_ APIs to fetch remote
audio files.

.. WARNING::
    Keep in mind that Zounds can't handle mp3 files yet.  
    Mp3 files will be skipped by 
    :py:class:`~zounds.acquire.acquirer.DiskAcquirer` and the 
    :code:`ingest.py` script.
'''
